import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from sources.bert.bert_dataset import BertCustomDataset


class BertClassifier:
    def __init__(self, n_classes=2, epochs=1, labels=None, model_save_path='/content/bert.pt'):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.labels = labels
        self.n_classes = n_classes

    def init_new_model(self, model_path, tokenizer_path):
        """Инициализация новой модели"""
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.n_classes
        )
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model.to(self.device)

    def load_model(self, model_path, tokenizer_path):
        """Загрузка сохраненной модели"""
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=self.n_classes)
        # загружаем веса
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.model.to(self.device)
        # Загружаем токенизатор
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print(f"Веса модели успешно загружены из {self.model_save_path}")

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = BertCustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = BertCustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=2, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=2, shuffle=False)

        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_loader) * self.epochs
        )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()  # Переводим модель в режим обучения
        losses = []  # Для сохранения потерь по батчам
        correct_predictions = 0  # Счётчик правильных предсказаний

        for data in self.train_loader:
            # Перенос данных на GPU (если используется)
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            # Forward pass: получаем выход модели
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Вычисляем предсказания (индекс максимального значения)
            preds = torch.argmax(outputs.logits, dim=1)

            # Вычисляем потерю
            loss = self.loss_fn(outputs.logits, targets)

            # Обновляем счётчик правильных ответов
            correct_predictions += torch.sum(preds == targets)

            # Сохраняем loss для усреднения
            losses.append(loss.item())

            # Backward pass и обновление весов
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Клиппинг градиентов
            self.optimizer.step()  # Обновление весов
            self.scheduler.step()  # Обновление learning rate
            self.optimizer.zero_grad()  # Очистка градиентов

        # Вычисляем среднюю точность и потерю по эпохе
        train_acc = correct_predictions.double() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model.eval()
        losses = []
        correct_predictions = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions.double() / len(self.valid_set)
        val_loss = np.mean(losses)

        # Вывод матрицы ошибок
        from sklearn.metrics import confusion_matrix
        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))

        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            # обучение модели
            train_acc, train_loss = self.fit()
            print(f'Train loss {train_loss} accuracy {train_acc}')

            # валидация модели
            val_acc, val_loss = self.eval()
            print(f'Val loss {val_loss} accuracy {val_acc}')
            print('-' * 10)

            # если точность стала выше, сохраняем веса в файл
            if val_acc > best_accuracy:
                torch.save(self.model.state_dict(), self.model_save_path)
                best_accuracy = val_acc

        # загружаем лучшие веса по окончанию обучения
        self.model.load_state_dict(torch.load(self.model_save_path))

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        if self.labels is None:
            return prediction  # Возвращаем индекс, если нет меток
        else:
            return self.labels[prediction]  # Возвращаем метку

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X : list or str
            List of texts or single text to classify

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
            Classes are ordered by lexicographic order.
        """
        self.model.eval()
        # Единообразная обработка ввода
        texts = [X] if isinstance(X, str) else X

        # Пакетная токенизация
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Предсказание
        with torch.no_grad():
            outputs = self.model(
                input_ids=encodings['input_ids'].to(self.device),
                attention_mask=encodings['attention_mask'].to(self.device)
            )

        # Применяем softmax для получения вероятностей
        probabilities = torch.softmax(outputs.logits, dim=1)

        # Конвертируем в numpy array
        return probabilities.cpu().numpy()
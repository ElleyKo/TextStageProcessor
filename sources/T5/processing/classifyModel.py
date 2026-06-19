import  torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
import numpy as np
import os
from . import disk
import pandas as pd
from datetime import datetime

class T5TextClassifyModel():
    general_loss = -1
    # название модели
    MODEL_NAME = 'SSD_HDD_v5.1'
    MODEL_NAME_2 = 'SSD_HDD_test_1'
    # загружать предобученную или загрузить свежую
    IS_LOAD_PRETRAINED = True
    # скачать свежую модель, сохранить и выйти
    IS_SAVE_ONLY = False
    # нужно ли собирать метрики для статистики  при обучении
    IS_TESTING_METRICS = False

    # сколько раз обучающие данные будут пропущены через модель
    TRAIN_EPOCHS = 200
    # на какое кол-во батчей будет делиться входной текст
    BATCH_COUNT = 1
    # кол-во символов в одном батчей
    BATCH_SYMBOLS = 1000
    # раз во сколько эпох проводить тестирование модели (если собираются метрики)
    TEST_EPOCHS = 10

    optimizer = None
    tokenizer = None
    model = None
    # данные для обучения
    pairs = None
    # данные для тестирования
    test_pairs = None

    train_data_df = None
    test_data_df = None
    def __init__(self):
        pass

    # Инициализирует модель
    # IS_LOAD_PRETRAINED = True - загружает модель с диска по MODEL_NAME
    # IS_LOAD_PRETRAINED = False - скачивает свежую модель с хаба
    def load_model(self, name: str = ''):
        print('LOAD_MODEL')
        print(f"CUDA available: {torch.cuda.is_available()}")

        if self.IS_LOAD_PRETRAINED:
            self.tokenizer = AutoTokenizer.from_pretrained(f"sources/T5/pretrained_model/{self.MODEL_NAME}")
            self.model = T5ForConditionalGeneration.from_pretrained(f"sources/T5/pretrained_model/{self.MODEL_NAME}")
            print(self.tokenizer)
        else:
            raw_model = 'cointegrated/rut5-base-multitask'
            disk.write_txt_log(f'START MODEL: {raw_model}')
            self.model = T5ForConditionalGeneration.from_pretrained(raw_model)
            self.tokenizer = AutoTokenizer.from_pretrained(raw_model)
            print(self.tokenizer)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        device = torch.device("cpu")
        self.model.to(device)

        print('MODEl INITIALIZED')

    # Загружает тренипровочные/тестовые данные в модель по указанному пути
    def load_data(self, folder: str):
        self.pairs = disk.load_test_input_txt(f'{os.getcwd()}/{folder}')
        self.pairs = disk.chop_texts(self.pairs, self.BATCH_SYMBOLS, self.BATCH_COUNT)
        print(f'DATA LOADED   -   pairs:{len(self.pairs)}')

    def load_data_pd(self, data_type:str = 'train', folder: str = ''):
        print(f'READ DATA: {folder}')
        df = disk.read_txt_files(folder)
        df['text'] = df['text'].str[:self.BATCH_SYMBOLS]

        if data_type == 'train':
            self.train_data_df = df
        else:
            self.test_data_df = df

    def train_model(self):
        # хранит данные для метрик
        lossList = []
        testingList = []

        print(f'train epochs: {self.TRAIN_EPOCHS}')
        disk.write_txt_log(f'TRAIN START')
        self.model.train()
        losses = []

        # кодируем вопрос и ответ
        x = self.tokenizer(self.train_data_df['text'].tolist(), return_tensors='pt', padding=True).to(self.model.device)
        y = self.tokenizer(self.train_data_df['label'].tolist(), return_tensors='pt', padding=True).to(self.model.device)
        # -100 - специальное значение, позволяющее не учитывать токены
        y.input_ids[y.input_ids == 0] = -100

        # df = pd.DataFrame(x, columns=['x1'])
        # df['target'] = y
        # print(df.head(5))

        report_steps = 1  # раз в сколько шагов печатаем результат
        for epoch in range(self.TRAIN_EPOCHS):
            print('EPOCH', epoch)
            #random.shuffle(self.pairs)
            # print(pairs)
            for i in [1]:
                #batch = self.pairs

                # вычисляем функцию потерь
                loss = self.model(
                    input_ids=x.input_ids,
                    attention_mask=x.attention_mask,
                    labels=y.input_ids,
                    decoder_attention_mask=y.attention_mask,
                    return_dict=True
                ).loss
                # делаем шаг градиентного спуска
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # печатаем скользящее среднее значение функции потерь
                losses.append(loss.item())
                loss_value = np.mean(losses[-report_steps:])
                if i % report_steps == 0:
                    print('step', i, 'loss', loss_value)

                # Если нужно замерить процесс обучения
                if self.IS_TESTING_METRICS:
                    # запись loss
                    lossList.append([epoch+1, loss_value])
                    disk.write_txt_log(f'epoch {epoch+1} loss: {loss_value}')
                    # запись тетсов (периодическая)
                    if (epoch + 1) % self.TEST_EPOCHS == 0:
                        predict_result = self.predict()
                        print('predict result:')
                        print(predict_result)
                        matches_count = predict_result[predict_result["label"] == predict_result["predict_result"]]
                        disk.write_txt_log(f'epoch {epoch + 1} Accuracy: {len(matches_count)}/{len(predict_result)}')
                        testingList.append([epoch+1, len(matches_count), len(predict_result)])
                        self.model.train()



        self.model.eval()

        self.model.save_pretrained(f"pretrained/{self.MODEL_NAME_2}")
        self.tokenizer.save_pretrained(f"pretrained/{self.MODEL_NAME_2}")
        disk.write_txt_log(f'model {self.MODEL_NAME_2} SAVED')
        disk.write_txt_log(f'TRAIN end')

        # Запись метрик
        if self.IS_TESTING_METRICS:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            loss_df = pd.DataFrame(lossList, columns=['epoch', 'loss'])
            testing_df = pd.DataFrame(testingList, columns=['epoch', 'match', 'all'])
            with pd.ExcelWriter(f"training_metrics_{timestamp}.xlsx") as writer:
                loss_df.to_excel(writer, sheet_name='loss', index=False)
                testing_df.to_excel(writer, sheet_name='tests', index=False)


    # классифицирует тексты
    def predict(self, answerAsList = False):
        answerList = []
        self.model.eval()

        def answer(x: str, **kwargs):
            inputs = self.tokenizer(x, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                hypotheses = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(hypotheses[0], skip_special_tokens=True)


        for index, row in self.test_data_df.iterrows():
            result = answer(row.text)
            answerList.append([row.filename, row.label, result])

        result_df = pd.DataFrame(answerList, columns=['filename', 'label', 'predict_result'])

        return result_df
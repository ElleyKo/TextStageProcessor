#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd
from PyQt5.QtCore import QObject
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from sources.bert.BertUtils import clean_text, lemmatize_text
from sources.bert.bert_classifier import BertClassifier
from sources.classification.clsf_util import collectFilesToDir, makeFileListLib
from sources.utils import clear_dir, Profiler


class BertClassificationCalculatorSignals(QObject):
    PrintInfo = pyqtSignal(str)
    Finished = pyqtSignal()
    UpdateProgressBar = pyqtSignal(int)
    UpdateMetricsTable = pyqtSignal(list)
    UpdateConfusionMatrix = pyqtSignal(tuple)


class BertClassificationCalculator(QThread):
    METHOD_BERT = 0

    def __init__(self, input_dir, output_dir, morph, configurations):
        super().__init__()
        self.profiler = Profiler()
        self.input_dir = input_dir + '/'
        self.output_dir = output_dir + '/classification/'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        clear_dir(self.output_dir)

        self.output_preprocessing_dir = self.output_dir + 'preprocessing/'
        self.first_call = True
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.morph = morph
        self.configurations = configurations
        self.texts = []
        self.categories = dict()
        self.signals = BertClassificationCalculatorSignals()
        self.need_preprocessing = False
        self.model_params = {}

        if len(self.input_dir) > 0 and self.input_dir[-1] == '/':
            self.input_dir = self.input_dir[:-1]
        last_slash_index = self.input_dir.rfind('/')
        self.input_dir_short = ''
        if last_slash_index != -1:
            self.input_dir_short = self.input_dir[last_slash_index + 1:]

    def run(self):
        self.signals.UpdateProgressBar.emit(0)

        self.method_input_dir = self.input_dir

        self.fdata, self.fclass, self.split, self.filenames = makeFileListLib(self.method_input_dir)

        self.signals.PrintInfo.emit("Файлы загружены из директорий")
        self.signals.UpdateProgressBar.emit(10)

        for index in range(len(self.filenames)):
            if '/' in self.filenames[index]:
                self.filenames[index] = self.filenames[index].split('/')[-1]

        self.test_filenames = self.filenames[self.split:]

        if self.method_index == BertClassificationCalculator.METHOD_BERT:
            self.classification_bert(self.morph)

        if self.first_call and self.need_preprocessing:
            self.first_call = False

        self.signals.UpdateProgressBar.emit(100)
        self.signals.PrintInfo.emit('Рассчеты закончены!')
        self.signals.Finished.emit()
        QApplication.restoreOverrideCursor()

    # Алгоритм BERT
    def classification_bert(self, morph):
        ##############PARAMS###################
        self.output_dir = self.output_dir + 'bert_out/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        input_dir = self.method_input_dir

        ###############ALGO##################
        testData, trainData = collectFilesToDir(input_dir)
        dfTest = pd.DataFrame(testData)
        dfTest = dfTest.reset_index(drop=True)
        print(dfTest)
        self.signals.PrintInfo.emit(dfTest.to_string())

        dfTrain = pd.DataFrame(trainData)
        dfTrain = dfTrain.reset_index(drop=True)
        self.signals.PrintInfo.emit(dfTrain.to_string())
        print(dfTrain)

        self.profiler.start()
        # Делаем препроцессинг
        if self.need_preprocessing:
            self.signals.UpdateProgressBar.emit(20)
            self.signals.PrintInfo.emit("Препроцессинг:")
            self.signals.PrintInfo.emit("1) Очистка текста")
            dfTest['prepare_text'] = dfTest['text'].apply(lambda text: clean_text(text))
            dfTrain['prepare_text'] = dfTrain['text'].apply(lambda text: clean_text(text))
            self.signals.UpdateProgressBar.emit(30)
            self.signals.PrintInfo.emit("2) Лемматизация")
            # применение лемматизации (приведение слов к первой форме)
            dfTest['text'] = dfTest['prepare_text'].apply(lambda text: lemmatize_text(text, morph=morph))
            dfTrain['text'] = dfTrain['prepare_text'].apply(lambda text: lemmatize_text(text, morph=morph))
        else:
            self.signals.PrintInfo.emit("Препроцессинг - пропускается")

        self.signals.PrintInfo.emit('Препроцессинг: выполнено за ' + self.profiler.stop() + ' с.')

        self.signals.UpdateProgressBar.emit(40)

        train_data = dfTrain
        test_data = dfTest

        countClasses = len(set(train_data.get('class')))

        self.training_model(self.spin_box_count_epoch, countClasses, train_data, test_data)

        self.signals.PrintInfo.emit('Выходные файлы: results.csv')

    def training_model(self, countEpoch, countClasses, train_data, test_data):
        save_dir = 'input_files/BERT'
        os.makedirs(save_dir, exist_ok=True)
        classifier = BertClassifier(
            n_classes=countClasses,
            epochs=countEpoch,
            labels=sorted(list(set(test_data['class']))),
            model_save_path=os.path.join(save_dir, 'bert.pt')
        )
        self.signals.PrintInfo.emit("Загрузка модели...")
        # Если не требуется обучение
        if self.model_params['not_train']:
            classifier.load_model(
                model_path='cointegrated/rubert-tiny',
                tokenizer_path='cointegrated/rubert-tiny'
            )
        elif self.model_params['fine_tune']:
            # Если требуется дообучение
            # Проверка наличия данных
            if len(train_data['text']) == 0:
                raise ValueError("Нет данных для дообучения!")
            classifier.load_model(
                model_path='cointegrated/rubert-tiny',
                tokenizer_path='cointegrated/rubert-tiny'
            )
        else:
            # Если требуется обучение
            # https://huggingface.co/cointegrated/rubert-tiny ruBERT
            # Проверка наличия данных
            if len(train_data['text']) == 0:
                raise ValueError("Нет данных для дообучения!")
            classifier.init_new_model(
                model_path="cointegrated/rubert-tiny",
                tokenizer_path="cointegrated/rubert-tiny"
            )

        classifier.preparation(
            X_train=list(train_data['text']),
            y_train=list(train_data['category']),
            X_valid=list(test_data['text']),
            y_valid=list(test_data['category'])
        )

        self.profiler.start()

        # Пропускаем этап обучения если не требуется
        if self.model_params['fine_tune'] or self.model_params['create_new']:
            self.signals.PrintInfo.emit("Обучение модели...")
            classifier.train()
            self.signals.PrintInfo.emit("Обучение завершено!")
        else:
            self.signals.PrintInfo.emit("Обучение не требуется")

        self.signals.UpdateProgressBar.emit(70)

        self.signals.PrintInfo.emit(
            'Обучение модели: выполнено за ' + self.profiler.stop() + ' с.')

        self.profiler.start()

        self.signals.PrintInfo.emit("Рассчет на тестовых данных...")
        # Выполняем классификацию
        texts = list(test_data['text'])
        labels = list(test_data['class'])
        predictions = [classifier.predict(t) for t in texts]

        self.signals.PrintInfo.emit(
            'Рассчет на тестовых данных: выполнено за ' + self.profiler.stop() + ' с.')


        precision, recall, f1 = precision_recall_fscore_support(labels, predictions, average='macro')[:3]
        # Получаем accuracy
        accuracy = accuracy_score(labels, predictions)

        # Отправка метрик
        self.update_metrics_table(accuracy, precision, recall, f1)
        # Для матрицы распределений (пример)
        cm = confusion_matrix(labels, predictions)
        self.update_confusion_matrix(cm, sorted(list(set(labels))))

        self.signals.UpdateProgressBar.emit(90)

        print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1score: {f1:.3f}')

        # Получить вероятности по всем классам
        probabilities = classifier.predict_proba(texts)  # список списков

        # Записать в текстовый файл
        write_results_to_file(
            output_filename=self.output_dir + 'results.csv',
            results=predictions,
            proba=probabilities,
            classes=set(labels),
            filenames=self.test_filenames
        )

    def set_model_params(self, params):
        """Устанавливает параметры модели"""
        self.model_params = params

    def update_metrics_table(self, accuracy, precision, recall, f1):
        self.signals.UpdateMetricsTable.emit([accuracy, precision, recall, f1])

    def update_confusion_matrix(self, matrix_data, classes):
        self.signals.UpdateConfusionMatrix.emit((matrix_data, classes))


def write_results_to_file(output_filename, results, proba, classes, filenames):
    result_lines = ["Результаты классификации:\n"]
    for index, result in enumerate(results):
        result_lines.append(f"Файл;{filenames[index]}")
        result_lines.append(f"Класс;{result}")
        result_lines.append("Вероятность:")
        for cl_index, cl in enumerate(classes):
            result_lines.append(f";{cl};{proba[index][cl_index]:.4f}")
        result_lines.append("")  # пустая строка между записями
    result_s = '\n'.join(result_lines)
    write_string_to_file(result_s, output_filename)


def write_string_to_file(content, output_filename):
    with open(output_filename, 'w', encoding='utf-8-sig') as f:
        f.write(content)

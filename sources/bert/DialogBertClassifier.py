#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from PyQt5 import QtCore, uic
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QDialog, QMessageBox,
                             QFileDialog, QTableWidgetItem)

from sources.bert.BertClassificationCalculator import BertClassificationCalculator
from sources.utils import Profiler


class DialogBertClassifier(QDialog):
    update_model_settings = pyqtSignal(dict)

    def __init__(self, dirname, morph, configurations, parent):
        super().__init__()
        uic.loadUi('sources/DialogBertClassifier.ui', self)

        flags = (Qt.Window | Qt.WindowSystemMenuHint |
                 Qt.WindowMinimizeButtonHint |
                 Qt.WindowMaximizeButtonHint |
                 Qt.WindowCloseButtonHint)
        self.setWindowFlags(flags)

        self.morph = morph
        self.configurations = configurations
        self.parent = parent
        self.input_dir = dirname
        self.model_path = ""
        self.lineEditInputDir.setText(dirname)

        # Инициализация обработчиков
        self.checkCreateModel.toggled.connect(self.toggle_model_creation)
        self.checkFineTune.toggled.connect(self.toggle_fine_tune)
        self.btnSelectModel.clicked.connect(self.select_model_file)

        # Начальное состояние
        self.checkCreateModel.setChecked(False)
        self.existingModelGroup.setVisible(True)
        self.profiler = Profiler()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.buttonClassify.clicked.connect(self.makeClassification)

        output_dir = self.configurations.get("output_files_directory", "output_files")
        self.calculator = BertClassificationCalculator(
            self.input_dir,
            output_dir,
            morph,
            self.configurations
        )
        self.calculator.signals.Finished.connect(self.onCalculationFinish)
        self.calculator.signals.UpdateProgressBar.connect(self.onUpdateProgressBar)
        self.calculator.signals.PrintInfo.connect(self.onTextLogAdd)
        self.output_dir = configurations.get("output_files_directory", "output_files/classification") + "/"
        self.calculator.signals.UpdateMetricsTable.connect(self.update_metrics_table)
        self.calculator.signals.UpdateConfusionMatrix.connect(self.update_confusion_matrix)

    # обработчик чекбокса создания новой модели
    def toggle_model_creation(self, checked):
        self.existingModelGroup.setVisible(not checked)
        self.checkFineTune.setEnabled(not checked)
        # спинбокс при обучении
        self.spin_box_count_epoch.setEnabled(checked)
        self.lineModelPath.setEnabled(not checked)
        self.btnSelectModel.setEnabled(not checked)

        if checked:
            self.lineModelPath.clear()
            self.checkFineTune.setChecked(False)
        else:
            # При отмене создания показываем актуальный путь
            if os.path.exists('input_files/BERT/bert.pt'):
                self.lineModelPath.setText(os.path.abspath('input_files/BERT/bert.pt'))

    # обработчик чекбокса дообучения
    def toggle_fine_tune(self, checked):
        # спинбокс при обучении
        self.spin_box_count_epoch.setEnabled(checked)
        self.checkCreateModel.setEnabled(not checked)
        if not checked:
            self.spin_box_count_epoch.setValue(0)

    def select_model_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            "",
            "Model Files (*.pt *.bin)"
        )
        if filename:
            self.lineModelPath.setText(filename)
            self.model_path = filename

    def onTextLogAdd(self, text):
        self.textEdit.append(text)
        self.repaint()

    def onUpdateProgressBar(self, value):
        self.progressBar.setValue(value)
        self.repaint()

    def onCalculationFinish(self):
        QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Внимание", "Процесс классификации завершен!")
        self.parent.show()

    def makeClassification(self):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.textEdit.clear()

            # Собираем параметры
            params = {
                'create_new': self.checkCreateModel.isChecked(),
                'fine_tune': self.checkFineTune.isChecked(),
                'model_path': self.lineModelPath.text(),
                'not_train': not self.checkCreateModel.isChecked() and not self.checkFineTune.isChecked(),
                'epochs': self.spin_box_count_epoch.value(),
                'fine_tune_epochs': self.spin_box_count_epoch.value(),
                'preprocess': self.checkBoxNeedPreprocessing.isChecked()
            }

            # Валидация параметров
            if not params['create_new'] and not os.path.exists(params['model_path']):
                raise FileNotFoundError("Файл модели не выбран или не существует")

            # Если не требуется дообучение - обнуляем эпохи
            if not params['fine_tune']:
                params['fine_tune_epochs'] = 0

            # Передаем параметры в калькулятор
            self.calculator.method_index = self.tabWidget.currentIndex()
            self.calculator.spin_box_count_epoch = self.spin_box_count_epoch.value()
            self.calculator.set_model_params(params)
            self.calculator.need_preprocessing = params['preprocess']

            self.calculator.start()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.checkBoxNeedPreprocessing.setEnabled(True)

    def closeEvent(self, event):
        self.parent.show()
        super().closeEvent(event)

    def update_metrics_table(self, metrics):
        """Обновление таблицы с метриками"""
        self.tableMetrics.setRowCount(1)
        for col, value in enumerate(metrics):
            item = QTableWidgetItem(f"{value:.3f}")
            self.tableMetrics.setItem(0, col, item)

    def update_confusion_matrix(self, data):
        """Обновление матрицы распределений"""
        matrix_data, classes = data
        classes = [str(c) for c in classes]  # Конвертируем в строки

        self.tableConfusionMatrix.setColumnCount(len(classes))
        self.tableConfusionMatrix.setRowCount(len(classes))

        # Заголовки
        self.tableConfusionMatrix.setHorizontalHeaderLabels(classes)
        self.tableConfusionMatrix.setVerticalHeaderLabels(classes)

        # Заполнение данных
        for i in range(len(classes)):
            for j in range(len(classes)):
                item = QTableWidgetItem(f"{matrix_data[i][j]:.2f}")
                self.tableConfusionMatrix.setItem(i, j, item)

    def toggle_fine_tune(self, checked):
        self.spin_box_count_epoch.setEnabled(checked)
        self.checkCreateModel.setEnabled(not checked)
        if not checked:
            self.spin_box_count_epoch.setValue(1)  # Устанавливаем минимальное значение вместо 0

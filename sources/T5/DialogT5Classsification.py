from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QWidget, QMainWindow
import pandas as pd
import os
from datetime import datetime


from .processing import classifyModel as cm
from ..utils import getDirFromUserSelection


class Ui_Form(QMainWindow):
    parentForm = None
    dirpath = None

    model = None
    modelAnswer = None

    def __init__(self, input_dir, dirpath, morph, configurations, parent):
        print('T5From init()')
        super().__init__()
        self.parentForm = parent
        self.parentForm = parent
        self.dirpath = dirpath
        self.inputDir = input_dir
        self.morph = morph
        self.configurations = configurations
        print(f'Filename: {self.dirpath}')
        self.setupUi(self)

        self.model = cm.T5TextClassifyModel()
        self.model.BATCH_SYMBOLS = 1000
        self.model.MODEL_PATH = 'SSD_HDD_v5'
        self.model.IS_LOAD_PRETRAINED = False
        if(self.model.IS_LOAD_PRETRAINED):
            self.model.load_model()
        self.model.load_data_pd(data_type='test', folder=self.dirpath)
        self.textEditFilesDisplay.setText(f"загружено файлов: {len(self.model.test_data_df)}")
        




    def setupUi(self, Form):
        print('T5FROM - setupUI()')
        self.setWindowTitle('Т5 классификация')
        Form.resize(520, 550)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)

        # Вывод по входным данным
        print('T5FROM - setupUI label')
        self.label = QtWidgets.QLabel("Файлы", self)
        self.label.setGeometry(10, 10, 200, 20)
        self.verticalLayout.addWidget(self.label)

        print('T5FROM - setupUI text_field')
        self.text_field = QtWidgets.QLineEdit(f'Путь: {self.dirpath}', self)
        self.text_field.setEnabled(False)  # Полностью неактивно
        self.text_field.setGeometry(10, 30, 500, 20)
        self.verticalLayout.addWidget(self.text_field)

        print('T5FROM - setupUI files_display')
        self.textEditFilesDisplay = QtWidgets.QLineEdit(parent=Form)
        self.textEditFilesDisplay.setGeometry(10, 50, 500, 20)
        #print(len(self.model.test_data_df))

        self.textEditFilesDisplay.setEnabled(False)  # Полностью неактивно
        self.verticalLayout.addWidget(self.textEditFilesDisplay)

        # Пуск
        self.buttonRun = QtWidgets.QPushButton("Запусить", self)
        self.buttonRun.setGeometry(10, 90, 100, 30)
        self.buttonRun.clicked.connect(self.boot_button_clicked)
        self.verticalLayout.addWidget(self.buttonRun)
        # self.button.clicked.connect(self.on_button)

        # Смена папки
        self.buttonFolder = QtWidgets.QPushButton("Сменить папку", self)
        self.buttonFolder.setGeometry(120, 90, 100, 30)
        self.buttonFolder.clicked.connect(self.folder_button_clicked)
        self.verticalLayout.addWidget(self.buttonFolder)

        # Сохранить в эксель
        self.buttonSave = QtWidgets.QPushButton("Сохранить результат на диск", self)
        self.buttonSave.setGeometry(10, 480, 300, 30)
        self.buttonSave.clicked.connect(self.button_save_clicked)
        self.buttonSave.setEnabled(False)
        self.verticalLayout.addWidget(self.buttonSave)

        # Результат
        print('T5FROM - setupUI result')
        self.label2 = QtWidgets.QLabel("Результат", self)
        self.label2.setGeometry(10, 130, 200, 30)
        self.verticalLayout.addWidget(self.label2)

        self.textEditOut = QtWidgets.QTextEdit(parent=Form)
        self.textEditOut.setGeometry(10, 170, 500, 300)
        self.textEditOut.setText("поле результата")
        self.verticalLayout.addWidget(self.textEditOut)

        print('T5FROM - setupUI end')

        #self.horizontalLayout = QtWidgets.QHBoxLayout()
        #self.verticalLayout.addLayout(self.horizontalLayout)

        QtCore.QMetaObject.connectSlotsByName(Form)

        print('T5FROM - setupUI ended')

    def boot_button_clicked(self):
        print('T5FROM - boot_button_clicked')
        answer = self.model.predict()
        self.modelAnswer = answer
        pd.set_option("display.max_rows", None)
        self.textEditOut.setText(str(answer[['filename', 'predict_result']]))
        print('T5FROM - boot_button_clicked ended')
        self.buttonSave.setEnabled(True)

    def folder_button_clicked(self):
        dirname = getDirFromUserSelection(self.dirpath)
        print(f'inputed dir name: {dirname}')
        if (dirname != None):
            self.dirpath = dirname
            self.model.load_data_pd(data_type='test', folder=self.dirpath)
            self.textEditFilesDisplay.setText(f"загружено файлов: {len(self.model.test_data_df)}")
            self.text_field.setText(f'Путь: {self.dirpath}')
            self.buttonSave.setEnabled(False)


    def button_save_clicked(self):
        dirname = getDirFromUserSelection(self.dirpath)
        if (dirname != None):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = f"{dirname}/T5_result_{timestamp}.xlsx"
            self.modelAnswer.to_excel(file_path, index=False)

    def closeEvent(self, event):
        if self.parent:
            self.parentForm.show()
        event.accept()
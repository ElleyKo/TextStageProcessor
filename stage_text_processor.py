#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymorphy2

from matplotlib import rc

import sys
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QSizePolicy, QAction, qApp, QSpacerItem, QApplication, QWidget, QFileDialog, QDialog, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore


from TextData import TextData
from TextPreprocessing import *
from TextClasterization import *
from TextClassification import *
from TextLSA import *


# Для корректного отображение шрифтов на графиках в Windows
if(os.name != 'posix'):
    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)


configurations = readConfigurationFile("configuration.cfg")
output_dir = configurations.get("output_files_directory", "output_files") + "/"

# Получаем экземпляр анализатора (10-20мб)
morph = pymorphy2.MorphAnalyzer()

class ProcessPlanSelectionWindow(QDialog):

    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):  
        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")

        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancelButton)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Выбор процесса')


# Класс главного окна
class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.texts = []


    def initUI(self):
        button_clasterization = QPushButton("Кластеризация")
        button_clasterization.setMinimumHeight(32)
        button_clasterization.clicked.connect(self.clasterization)

        button_classification = QPushButton("Классификация")
        button_classification.setMinimumHeight(32)
        button_classification.clicked.connect(self.classification)

        button_lsa = QPushButton("Латентно-семантический анализ")
        button_lsa.setMinimumHeight(32)
        button_lsa.clicked.connect(self.makeLSA)

        spacer = QSpacerItem(20,40,QSizePolicy.Minimum,QSizePolicy.Expanding)

        vbox = QVBoxLayout()
        vbox.addWidget(button_clasterization)
        vbox.addWidget(button_classification)
        vbox.addWidget(button_lsa)
        vbox.addItem(spacer)

        widget = QWidget();
        widget.setLayout(vbox);
        self.setCentralWidget(widget);
        self.setGeometry(300, 300, 480, 320)
        self.setWindowTitle('Этапный текстовый процессор')    
        self.show()


    def getFilenamesFromUserSelection(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Открыть файлы для анализа", "", "Text Files (*.txt)", None)
        if(len(filenames) > 0):
            return filenames
        else:
            return None


    def clasterization(self):
        print("Кластеризация")
        filenames = self.getFilenamesFromUserSelection()
        if(filenames != None):
            makeClasterization(filenames, morph, configurations)
        print("Кластеризация - завершено")
        QMessageBox.information(self,"Внимание", "Процесс завершен!")


    def classification(self):
        print("Классификация")
        filenames = self.getFilenamesFromUserSelection()
        if(filenames != None):
            makeClassification(filenames)
        print("Классификация - завершено")
        QMessageBox.information(self,"Внимание", "Процесс завершен!")

    def makeLSA(self):
        print("LSA")
        filenames = self.getFilenamesFromUserSelection()
        if(filenames != None):
            makeLSA(filenames, morph, configurations)
        print("LSA - завершено")
        QMessageBox.information(self,"Внимание", "Процесс завершен!")



    def buttonOpenClicked(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Открыть файлы для анализа", "", "Text Files (*.txt)", None)
        if(len(filenames) > 0):
            self.texts = loadInputFilesFromList(filenames)

            for text in self.texts:
                print(text.filename)

            planSelectorWindow = ProcessPlanSelectionWindow()
            planSelectorWindow.show()
            planSelectorWindow.setAttribute(QtCore.Qt.WA_DeleteOnClose)

            result = planSelectorWindow.exec_()
            if(result == QDialog.Accepted):
                print("Производим анализ текстов...")
                
            else:
                self.texts = []


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())




# ### ПРОГРАММА  ---------------------------------------------------------------
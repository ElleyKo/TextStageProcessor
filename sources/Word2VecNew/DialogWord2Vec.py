# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DialogWord2Vec.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Word2VecDialog(object):
    def setupUi(self, Word2VecDialog):
        Word2VecDialog.setObjectName("Word2VecDialog")
        Word2VecDialog.setWindowModality(QtCore.Qt.NonModal)
        Word2VecDialog.setEnabled(True)
        Word2VecDialog.resize(1027, 614)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Word2VecDialog.sizePolicy().hasHeightForWidth())
        Word2VecDialog.setSizePolicy(sizePolicy)
        Word2VecDialog.setMinimumSize(QtCore.QSize(657, 580))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        Word2VecDialog.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("img/w2v_image.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Word2VecDialog.setWindowIcon(QtGui.QIcon(icon))
        Word2VecDialog.setModal(False)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Word2VecDialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mainAVLaout = QtWidgets.QVBoxLayout()
        self.mainAVLaout.setObjectName("mainAVLaout")
        self.tabWidget = QtWidgets.QTabWidget(Word2VecDialog)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.modelCreatorTab = QtWidgets.QWidget()
        self.modelCreatorTab.setObjectName("modelCreatorTab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.modelCreatorTab)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.createMainHLayout = QtWidgets.QHBoxLayout()
        self.createMainHLayout.setObjectName("createMainHLayout")
        self.createVLayout = QtWidgets.QVBoxLayout()
        self.createVLayout.setObjectName("createVLayout")
        self.word2VecConfigGBox = QtWidgets.QGroupBox(self.modelCreatorTab)
        self.word2VecConfigGBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.word2VecConfigGBox.sizePolicy().hasHeightForWidth())
        self.word2VecConfigGBox.setSizePolicy(sizePolicy)
        self.word2VecConfigGBox.setMinimumSize(QtCore.QSize(300, 100))
        self.word2VecConfigGBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.word2VecConfigGBox.setObjectName("word2VecConfigGBox")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.word2VecConfigGBox)
        self.verticalLayout_8.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_8.setContentsMargins(9, 8, -1, -1)
        self.verticalLayout_8.setSpacing(6)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.filePathHLayout = QtWidgets.QHBoxLayout()
        self.filePathHLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.filePathHLayout.setObjectName("filePathHLayout")
        self.selectAnotherPathBtn = QtWidgets.QPushButton(self.word2VecConfigGBox)
        self.selectAnotherPathBtn.setMinimumSize(QtCore.QSize(100, 30))
        self.selectAnotherPathBtn.setObjectName("selectAnotherPathBtn")
        self.filePathHLayout.addWidget(self.selectAnotherPathBtn, 0, QtCore.Qt.AlignLeft)
        self.filePathLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filePathLbl.sizePolicy().hasHeightForWidth())
        self.filePathLbl.setSizePolicy(sizePolicy)
        self.filePathLbl.setMinimumSize(QtCore.QSize(191, 30))
        self.filePathLbl.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.filePathLbl.setWordWrap(True)
        self.filePathLbl.setObjectName("filePathLbl")
        self.filePathHLayout.addWidget(self.filePathLbl, 0, QtCore.Qt.AlignHCenter)
        self.filePathField = QtWidgets.QLineEdit(self.word2VecConfigGBox)
        self.filePathField.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filePathField.sizePolicy().hasHeightForWidth())
        self.filePathField.setSizePolicy(sizePolicy)
        self.filePathField.setMinimumSize(QtCore.QSize(0, 20))
        self.filePathField.setMaximumSize(QtCore.QSize(16777215, 63))
        self.filePathField.setReadOnly(True)
        self.filePathField.setObjectName("filePathField")
        self.filePathHLayout.addWidget(self.filePathField)
        self.verticalLayout_8.addLayout(self.filePathHLayout)
        self.wordFrequencyHLayout = QtWidgets.QHBoxLayout()
        self.wordFrequencyHLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.wordFrequencyHLayout.setObjectName("wordFrequencyHLayout")
        self.wordFrequencyLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.wordFrequencyLbl.setWordWrap(True)
        self.wordFrequencyLbl.setObjectName("wordFrequencyLbl")
        self.wordFrequencyHLayout.addWidget(self.wordFrequencyLbl)
        self.wordFrequencyField = QtWidgets.QSpinBox(self.word2VecConfigGBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.wordFrequencyField.sizePolicy().hasHeightForWidth())
        self.wordFrequencyField.setSizePolicy(sizePolicy)
        self.wordFrequencyField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.wordFrequencyField.setMinimum(1)
        self.wordFrequencyField.setMaximum(150)
        self.wordFrequencyField.setProperty("value", 5)
        self.wordFrequencyField.setObjectName("wordFrequencyField")
        self.wordFrequencyHLayout.addWidget(self.wordFrequencyField)
        self.verticalLayout_8.addLayout(self.wordFrequencyHLayout)
        self.vectorSizeHLayout = QtWidgets.QHBoxLayout()
        self.vectorSizeHLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.vectorSizeHLayout.setObjectName("vectorSizeHLayout")
        self.vectorSizeLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.vectorSizeLbl.setWordWrap(True)
        self.vectorSizeLbl.setObjectName("vectorSizeLbl")
        self.vectorSizeHLayout.addWidget(self.vectorSizeLbl)
        self.vectorSizeField = QtWidgets.QSpinBox(self.word2VecConfigGBox)
        self.vectorSizeField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.vectorSizeField.setMinimum(1)
        self.vectorSizeField.setMaximum(100000)
        self.vectorSizeField.setProperty("value", 100)
        self.vectorSizeField.setObjectName("vectorSizeField")
        self.vectorSizeHLayout.addWidget(self.vectorSizeField)
        self.verticalLayout_8.addLayout(self.vectorSizeHLayout)
        self.trainingSpeedHLayout = QtWidgets.QHBoxLayout()
        self.trainingSpeedHLayout.setObjectName("trainingSpeedHLayout")
        self.trainingSpeedLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.trainingSpeedLbl.setWordWrap(True)
        self.trainingSpeedLbl.setObjectName("trainingSpeedLbl")
        self.trainingSpeedHLayout.addWidget(self.trainingSpeedLbl)
        self.trainingSpeedField = QtWidgets.QDoubleSpinBox(self.word2VecConfigGBox)
        self.trainingSpeedField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.trainingSpeedField.setDecimals(3)
        self.trainingSpeedField.setMinimum(0.001)
        self.trainingSpeedField.setMaximum(1.0)
        self.trainingSpeedField.setProperty("value", 0.025)
        self.trainingSpeedField.setObjectName("trainingSpeedField")
        self.trainingSpeedHLayout.addWidget(self.trainingSpeedField)
        self.verticalLayout_8.addLayout(self.trainingSpeedHLayout)
        self.negativeSamplingHLayout = QtWidgets.QHBoxLayout()
        self.negativeSamplingHLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.negativeSamplingHLayout.setObjectName("negativeSamplingHLayout")
        self.negativeSamplingLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.negativeSamplingLbl.setMinimumSize(QtCore.QSize(155, 20))
        self.negativeSamplingLbl.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.negativeSamplingLbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.negativeSamplingLbl.setWordWrap(True)
        self.negativeSamplingLbl.setObjectName("negativeSamplingLbl")
        self.negativeSamplingHLayout.addWidget(self.negativeSamplingLbl)
        self.negativeSamplingField = QtWidgets.QSpinBox(self.word2VecConfigGBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.negativeSamplingField.sizePolicy().hasHeightForWidth())
        self.negativeSamplingField.setSizePolicy(sizePolicy)
        self.negativeSamplingField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.negativeSamplingField.setMinimum(1)
        self.negativeSamplingField.setMaximum(20)
        self.negativeSamplingField.setProperty("value", 5)
        self.negativeSamplingField.setObjectName("negativeSamplingField")
        self.negativeSamplingHLayout.addWidget(self.negativeSamplingField)
        self.verticalLayout_8.addLayout(self.negativeSamplingHLayout)
        self.negativeSamplingExpHLayout = QtWidgets.QHBoxLayout()
        self.negativeSamplingExpHLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.negativeSamplingExpHLayout.setObjectName("negativeSamplingExpHLayout")
        self.negativeSamplingExpLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.negativeSamplingExpLbl.setMinimumSize(QtCore.QSize(155, 20))
        self.negativeSamplingExpLbl.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.negativeSamplingExpLbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.negativeSamplingExpLbl.setWordWrap(True)
        self.negativeSamplingExpLbl.setObjectName("negativeSamplingExpLbl")
        self.negativeSamplingExpHLayout.addWidget(self.negativeSamplingExpLbl)
        self.negativeSamplingExpField = QtWidgets.QDoubleSpinBox(self.word2VecConfigGBox)
        self.negativeSamplingExpField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.negativeSamplingExpField.setDecimals(3)
        self.negativeSamplingExpField.setMinimum(0.001)
        self.negativeSamplingExpField.setMaximum(1.0)
        self.negativeSamplingExpField.setProperty("value", 0.075)
        self.negativeSamplingExpField.setObjectName("negativeSamplingExpField")
        self.negativeSamplingExpHLayout.addWidget(self.negativeSamplingExpField)
        self.verticalLayout_8.addLayout(self.negativeSamplingExpHLayout)
        self.windowHLayout = QtWidgets.QHBoxLayout()
        self.windowHLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.windowHLayout.setObjectName("windowHLayout")
        self.windowLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.windowLbl.setMinimumSize(QtCore.QSize(155, 20))
        self.windowLbl.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.windowLbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.windowLbl.setWordWrap(True)
        self.windowLbl.setObjectName("windowLbl")
        self.windowHLayout.addWidget(self.windowLbl)
        self.windowField = QtWidgets.QSpinBox(self.word2VecConfigGBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.windowField.sizePolicy().hasHeightForWidth())
        self.windowField.setSizePolicy(sizePolicy)
        self.windowField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.windowField.setMinimum(1)
        self.windowField.setMaximum(150)
        self.windowField.setProperty("value", 5)
        self.windowField.setObjectName("windowField")
        self.windowHLayout.addWidget(self.windowField)
        self.verticalLayout_8.addLayout(self.windowHLayout)
        self.algorithmHLayout = QtWidgets.QHBoxLayout()
        self.algorithmHLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.algorithmHLayout.setSpacing(6)
        self.algorithmHLayout.setObjectName("algorithmHLayout")
        self.algorithmRadioLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.algorithmRadioLbl.setMinimumSize(QtCore.QSize(150, 20))
        self.algorithmRadioLbl.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.algorithmRadioLbl.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.algorithmRadioLbl.setWordWrap(True)
        self.algorithmRadioLbl.setObjectName("algorithmRadioLbl")
        self.algorithmHLayout.addWidget(self.algorithmRadioLbl)
        self.algorithmsGroupBox = QtWidgets.QGroupBox(self.word2VecConfigGBox)
        self.algorithmsGroupBox.setMinimumSize(QtCore.QSize(200, 0))
        self.algorithmsGroupBox.setMaximumSize(QtCore.QSize(110, 16777215))
        self.algorithmsGroupBox.setAutoFillBackground(False)
        self.algorithmsGroupBox.setTitle("")
        self.algorithmsGroupBox.setFlat(False)
        self.algorithmsGroupBox.setCheckable(False)
        self.algorithmsGroupBox.setObjectName("algorithmsGroupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.algorithmsGroupBox)
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_2.setContentsMargins(1, -1, -1, -1)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.skipGramRadio = QtWidgets.QRadioButton(self.algorithmsGroupBox)
        self.skipGramRadio.setMinimumSize(QtCore.QSize(120, 20))
        self.skipGramRadio.setMaximumSize(QtCore.QSize(100, 16777215))
        self.skipGramRadio.setObjectName("skipGramRadio")
        self.horizontalLayout_2.addWidget(self.skipGramRadio)
        self.CBOWRadio = QtWidgets.QRadioButton(self.algorithmsGroupBox)
        self.CBOWRadio.setMinimumSize(QtCore.QSize(0, 20))
        self.CBOWRadio.setMaximumSize(QtCore.QSize(110, 16777215))
        self.CBOWRadio.setChecked(True)
        self.CBOWRadio.setObjectName("CBOWRadio")
        self.horizontalLayout_2.addWidget(self.CBOWRadio)
        self.algorithmHLayout.addWidget(self.algorithmsGroupBox)
        self.verticalLayout_8.addLayout(self.algorithmHLayout)
        self.epochNumberHLayout = QtWidgets.QHBoxLayout()
        self.epochNumberHLayout.setObjectName("epochNumberHLayout")
        self.epochNumberLbl = QtWidgets.QLabel(self.word2VecConfigGBox)
        self.epochNumberLbl.setWordWrap(True)
        self.epochNumberLbl.setObjectName("epochNumberLbl")
        self.epochNumberHLayout.addWidget(self.epochNumberLbl)
        self.epochNumberField = QtWidgets.QSpinBox(self.word2VecConfigGBox)
        self.epochNumberField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.epochNumberField.setMinimum(1)
        self.epochNumberField.setMaximum(1000)
        self.epochNumberField.setProperty("value", 5)
        self.epochNumberField.setObjectName("epochNumberField")
        self.epochNumberHLayout.addWidget(self.epochNumberField)
        self.verticalLayout_8.addLayout(self.epochNumberHLayout)
        self.nounOnlyHLayout = QtWidgets.QHBoxLayout()
        self.nounOnlyHLayout.setObjectName("nounOnlyHLayout")
        self.nounOnlyCheck = QtWidgets.QCheckBox(self.word2VecConfigGBox)
        self.nounOnlyCheck.setChecked(True)
        self.nounOnlyCheck.setObjectName("nounOnlyCheck")
        self.nounOnlyHLayout.addWidget(self.nounOnlyCheck)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.nounOnlyHLayout.addItem(spacerItem)
        self.verticalLayout_8.addLayout(self.nounOnlyHLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_8.addItem(spacerItem1)
        self.createVLayout.addWidget(self.word2VecConfigGBox)
        self.createModelBar = QtWidgets.QProgressBar(self.modelCreatorTab)
        self.createModelBar.setProperty("value", 0)
        self.createModelBar.setTextVisible(True)
        self.createModelBar.setObjectName("createModelBar")
        self.createVLayout.addWidget(self.createModelBar)
        self.createModelBtn = QtWidgets.QPushButton(self.modelCreatorTab)
        self.createModelBtn.setObjectName("createModelBtn")
        self.createVLayout.addWidget(self.createModelBtn)
        self.recreateModelBtn = QtWidgets.QPushButton(self.modelCreatorTab)
        self.recreateModelBtn.setObjectName("recreateModelBtn")
        self.createVLayout.addWidget(self.recreateModelBtn)
        self.createMainHLayout.addLayout(self.createVLayout)
        self.createLogTextEdit = QtWidgets.QTextEdit(self.modelCreatorTab)
        self.createLogTextEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.createLogTextEdit.sizePolicy().hasHeightForWidth())
        self.createLogTextEdit.setSizePolicy(sizePolicy)
        self.createLogTextEdit.setMinimumSize(QtCore.QSize(350, 0))
        self.createLogTextEdit.setReadOnly(True)
        self.createLogTextEdit.setObjectName("createLogTextEdit")
        self.createMainHLayout.addWidget(self.createLogTextEdit)
        self.verticalLayout_4.addLayout(self.createMainHLayout)
        self.tabWidget.addTab(self.modelCreatorTab, "")
        self.tSNEVisualizeTab = QtWidgets.QWidget()
        self.tSNEVisualizeTab.setObjectName("tSNEVisualizeTab")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.tSNEVisualizeTab)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.visualizeHLayout = QtWidgets.QHBoxLayout()
        self.visualizeHLayout.setObjectName("visualizeHLayout")
        self.visualAlgorithmVLayout = QtWidgets.QVBoxLayout()
        self.visualAlgorithmVLayout.setObjectName("visualAlgorithmVLayout")
        self.selectModelGBox = QtWidgets.QGroupBox(self.tSNEVisualizeTab)
        self.selectModelGBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.selectModelGBox.sizePolicy().hasHeightForWidth())
        self.selectModelGBox.setSizePolicy(sizePolicy)
        self.selectModelGBox.setMinimumSize(QtCore.QSize(300, 80))
        self.selectModelGBox.setMaximumSize(QtCore.QSize(16777215, 65))
        self.selectModelGBox.setObjectName("selectModelGBox")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.selectModelGBox)
        self.verticalLayout_10.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_10.setContentsMargins(9, 8, -1, -1)
        self.verticalLayout_10.setSpacing(6)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.selectModelHLayout = QtWidgets.QHBoxLayout()
        self.selectModelHLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.selectModelHLayout.setObjectName("selectModelHLayout")
        self.selectModelField = QtWidgets.QLineEdit(self.selectModelGBox)
        self.selectModelField.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.selectModelField.sizePolicy().hasHeightForWidth())
        self.selectModelField.setSizePolicy(sizePolicy)
        self.selectModelField.setMinimumSize(QtCore.QSize(348, 30))
        self.selectModelField.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.selectModelField.setReadOnly(True)
        self.selectModelField.setObjectName("selectModelField")
        self.selectModelHLayout.addWidget(self.selectModelField)
        self.selectModelBtn = QtWidgets.QPushButton(self.selectModelGBox)
        self.selectModelBtn.setMinimumSize(QtCore.QSize(110, 30))
        self.selectModelBtn.setObjectName("selectModelBtn")
        self.selectModelHLayout.addWidget(self.selectModelBtn)
        self.verticalLayout_10.addLayout(self.selectModelHLayout)
        self.visualAlgorithmVLayout.addWidget(self.selectModelGBox)
        self.plotBtnHLayout = QtWidgets.QHBoxLayout()
        self.plotBtnHLayout.setObjectName("plotBtnHLayout")
        self.visualAlgorithmVLayout.addLayout(self.plotBtnHLayout)
        self.visualizeGBox = QtWidgets.QGroupBox(self.tSNEVisualizeTab)
        self.visualizeGBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.visualizeGBox.sizePolicy().hasHeightForWidth())
        self.visualizeGBox.setSizePolicy(sizePolicy)
        self.visualizeGBox.setMinimumSize(QtCore.QSize(500, 63))
        self.visualizeGBox.setMaximumSize(QtCore.QSize(16777215, 63))
        self.visualizeGBox.setObjectName("visualizeGBox")
        self.btnHLayout = QtWidgets.QHBoxLayout(self.visualizeGBox)
        self.btnHLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.btnHLayout.setObjectName("btnHLayout")
        self.pcaTSNERadioBtn = QtWidgets.QRadioButton(self.visualizeGBox)
        self.pcaTSNERadioBtn.setMinimumSize(QtCore.QSize(120, 30))
        self.pcaTSNERadioBtn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pcaTSNERadioBtn.setChecked(True)
        self.pcaTSNERadioBtn.setObjectName("pcaTSNERadioBtn")
        self.btnHLayout.addWidget(self.pcaTSNERadioBtn)
        self.tsvdTSNERadioBtn = QtWidgets.QRadioButton(self.visualizeGBox)
        self.tsvdTSNERadioBtn.setMinimumSize(QtCore.QSize(178, 30))
        self.tsvdTSNERadioBtn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.tsvdTSNERadioBtn.setObjectName("tsvdTSNERadioBtn")
        self.btnHLayout.addWidget(self.tsvdTSNERadioBtn)
        self.tsvdRadioBtn = QtWidgets.QRadioButton(self.visualizeGBox)
        self.tsvdRadioBtn.setMinimumSize(QtCore.QSize(120, 30))
        self.tsvdRadioBtn.setMaximumSize(QtCore.QSize(110, 16777215))
        self.tsvdRadioBtn.setChecked(False)
        self.tsvdRadioBtn.setObjectName("tsvdRadioBtn")
        self.btnHLayout.addWidget(self.tsvdRadioBtn)
        self.pcaRadioBtn = QtWidgets.QRadioButton(self.visualizeGBox)
        self.pcaRadioBtn.setMinimumSize(QtCore.QSize(60, 30))
        self.pcaRadioBtn.setMaximumSize(QtCore.QSize(100, 16777215))
        self.pcaRadioBtn.setObjectName("pcaRadioBtn")
        self.btnHLayout.addWidget(self.pcaRadioBtn)
        self.visualAlgorithmVLayout.addWidget(self.visualizeGBox)
        self.searchOnlyHLayout = QtWidgets.QHBoxLayout()
        self.searchOnlyHLayout.setObjectName("searchOnlyHLayout")
        self.searchOnlyCheck = QtWidgets.QCheckBox(self.tSNEVisualizeTab)
        self.searchOnlyCheck.setEnabled(False)
        self.searchOnlyCheck.setChecked(False)
        self.searchOnlyCheck.setObjectName("searchOnlyCheck")
        self.searchOnlyHLayout.addWidget(self.searchOnlyCheck)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.searchOnlyHLayout.addItem(spacerItem2)
        self.visualAlgorithmVLayout.addLayout(self.searchOnlyHLayout)
        self.visualizeBtnHLayout = QtWidgets.QHBoxLayout()
        self.visualizeBtnHLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.visualizeBtnHLayout.setSpacing(6)
        self.visualizeBtnHLayout.setObjectName("visualizeBtnHLayout")
        self.visualizeBtn = QtWidgets.QPushButton(self.tSNEVisualizeTab)
        self.visualizeBtn.setObjectName("visualizeBtn")
        self.visualizeBtnHLayout.addWidget(self.visualizeBtn)
        self.clearBtn = QtWidgets.QPushButton(self.tSNEVisualizeTab)
        self.clearBtn.setObjectName("clearBtn")
        self.visualizeBtnHLayout.addWidget(self.clearBtn)
        self.visualAlgorithmVLayout.addLayout(self.visualizeBtnHLayout)
        self.plotVLayout = QtWidgets.QVBoxLayout()
        self.plotVLayout.setObjectName("plotVLayout")
        self.plotWidget = QtWidgets.QWidget(self.tSNEVisualizeTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plotWidget.sizePolicy().hasHeightForWidth())
        self.plotWidget.setSizePolicy(sizePolicy)
        self.plotWidget.setMinimumSize(QtCore.QSize(400, 0))
        self.plotWidget.setObjectName("plotWidget")
        self.mplvl_2 = QtWidgets.QVBoxLayout(self.plotWidget)
        self.mplvl_2.setContentsMargins(2, 2, 2, 2)
        self.mplvl_2.setSpacing(2)
        self.mplvl_2.setObjectName("mplvl_2")
        self.plotVLayout.addWidget(self.plotWidget)
        self.visualAlgorithmVLayout.addLayout(self.plotVLayout)
        self.visualizeHLayout.addLayout(self.visualAlgorithmVLayout)
        self.searchVLayout = QtWidgets.QVBoxLayout()
        self.searchVLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.searchVLayout.setContentsMargins(5, 0, 5, 5)
        self.searchVLayout.setSpacing(6)
        self.searchVLayout.setObjectName("searchVLayout")
        self.searchQueryGBox = QtWidgets.QGroupBox(self.tSNEVisualizeTab)
        self.searchQueryGBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.searchQueryGBox.sizePolicy().hasHeightForWidth())
        self.searchQueryGBox.setSizePolicy(sizePolicy)
        self.searchQueryGBox.setMinimumSize(QtCore.QSize(300, 110))
        self.searchQueryGBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.searchQueryGBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.searchQueryGBox.setObjectName("searchQueryGBox")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.searchQueryGBox)
        self.verticalLayout_9.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_9.setContentsMargins(9, 8, -1, 9)
        self.verticalLayout_9.setSpacing(6)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.topNHLayout = QtWidgets.QHBoxLayout()
        self.topNHLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.topNHLayout.setObjectName("topNHLayout")
        self.topNLbl = QtWidgets.QLabel(self.searchQueryGBox)
        self.topNLbl.setWordWrap(True)
        self.topNLbl.setObjectName("topNLbl")
        self.topNHLayout.addWidget(self.topNLbl)
        self.topNField = QtWidgets.QSpinBox(self.searchQueryGBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.topNField.sizePolicy().hasHeightForWidth())
        self.topNField.setSizePolicy(sizePolicy)
        self.topNField.setMaximumSize(QtCore.QSize(110, 16777215))
        self.topNField.setMinimum(1)
        self.topNField.setMaximum(150)
        self.topNField.setProperty("value", 15)
        self.topNField.setObjectName("topNField")
        self.topNHLayout.addWidget(self.topNField)
        self.verticalLayout_9.addLayout(self.topNHLayout)
        self.searchQueryHLayout = QtWidgets.QHBoxLayout()
        self.searchQueryHLayout.setObjectName("searchQueryHLayout")
        self.searchQueryField = QtWidgets.QLineEdit(self.searchQueryGBox)
        self.searchQueryField.setMinimumSize(QtCore.QSize(0, 30))
        self.searchQueryField.setObjectName("searchQueryField")
        self.searchQueryHLayout.addWidget(self.searchQueryField)
        self.searchQueryBtn = QtWidgets.QPushButton(self.searchQueryGBox)
        self.searchQueryBtn.setMinimumSize(QtCore.QSize(0, 30))
        self.searchQueryBtn.setObjectName("searchQueryBtn")
        self.searchQueryHLayout.addWidget(self.searchQueryBtn)
        self.verticalLayout_9.addLayout(self.searchQueryHLayout)
        self.searchVLayout.addWidget(self.searchQueryGBox)
        self.visualizeLogTextEdit = QtWidgets.QTextEdit(self.tSNEVisualizeTab)
        self.visualizeLogTextEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.visualizeLogTextEdit.sizePolicy().hasHeightForWidth())
        self.visualizeLogTextEdit.setSizePolicy(sizePolicy)
        self.visualizeLogTextEdit.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.visualizeLogTextEdit.setReadOnly(True)
        self.visualizeLogTextEdit.setObjectName("visualizeLogTextEdit")
        self.searchVLayout.addWidget(self.visualizeLogTextEdit)
        self.visualizeHLayout.addLayout(self.searchVLayout)
        self.verticalLayout_5.addLayout(self.visualizeHLayout)
        self.tabWidget.addTab(self.tSNEVisualizeTab, "")
        self.mainAVLaout.addWidget(self.tabWidget)
        self.horizontalLayout.addLayout(self.mainAVLaout)

        self.retranslateUi(Word2VecDialog)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Word2VecDialog)

    def retranslateUi(self, Word2VecDialog):
        _translate = QtCore.QCoreApplication.translate
        Word2VecDialog.setWindowTitle(_translate("Word2VecDialog", "Алгоритм word2vec"))
        self.word2VecConfigGBox.setTitle(_translate("Word2VecDialog", "Параметры модели"))
        self.selectAnotherPathBtn.setText(_translate("Word2VecDialog", "Изменить"))
        self.filePathLbl.setText(_translate("Word2VecDialog", "Входные данные (input file)"))
        self.filePathField.setText(_translate("Word2VecDialog", "input_files/Word2Vec"))
        self.wordFrequencyLbl.setText(_translate("Word2VecDialog", "Количество повторений слова (min_count)"))
        self.vectorSizeLbl.setText(_translate("Word2VecDialog", "Размерность вектора слова (size)"))
        self.trainingSpeedLbl.setText(_translate("Word2VecDialog", "Коэффициент скорости обучения (learning rate)"))
        self.negativeSamplingLbl.setText(_translate("Word2VecDialog", "Negative sampling"))
        self.negativeSamplingExpLbl.setText(_translate("Word2VecDialog", "Negative sampling exponent"))
        self.windowLbl.setText(_translate("Word2VecDialog", "Размер окна (window)"))
        self.algorithmRadioLbl.setText(_translate("Word2VecDialog", "Алгоритм тренировки модели (train algorithm)"))
        self.skipGramRadio.setText(_translate("Word2VecDialog", "skip-gram"))
        self.CBOWRadio.setText(_translate("Word2VecDialog", "CBOW"))
        self.epochNumberLbl.setText(_translate("Word2VecDialog", "Количество эпох (epoch)"))
        self.nounOnlyCheck.setText(_translate("Word2VecDialog", "Только существительные"))
        self.createModelBar.setFormat(_translate("Word2VecDialog", "%p%"))
        self.createModelBtn.setText(_translate("Word2VecDialog", "Создать модель"))
        self.recreateModelBtn.setText(_translate("Word2VecDialog", "Пересоздать модель"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.modelCreatorTab), _translate("Word2VecDialog", "Создание модели"))
        self.selectModelGBox.setTitle(_translate("Word2VecDialog", "Загрузить модель"))
        self.selectModelField.setText(_translate("Word2VecDialog", "..."))
        self.selectModelBtn.setText(_translate("Word2VecDialog", "Выбрать модель"))
        self.visualizeGBox.setTitle(_translate("Word2VecDialog", "Алгоритмы визуализации"))
        self.pcaTSNERadioBtn.setText(_translate("Word2VecDialog", "PCA и t-SNE"))
        self.tsvdTSNERadioBtn.setText(_translate("Word2VecDialog", " TruncatedSVD и t-SNE"))
        self.tsvdRadioBtn.setText(_translate("Word2VecDialog", " TruncatedSVD"))
        self.pcaRadioBtn.setText(_translate("Word2VecDialog", "PCA"))
        self.searchOnlyCheck.setText(_translate("Word2VecDialog", "Отобразить только результаты поиска"))
        self.visualizeBtn.setText(_translate("Word2VecDialog", "Визуализировать"))
        self.clearBtn.setText(_translate("Word2VecDialog", "Очистить область"))
        self.searchQueryGBox.setTitle(_translate("Word2VecDialog", "Поиск слова"))
        self.topNLbl.setText(_translate("Word2VecDialog", "Количество похожих слов в выборке (top n)"))
        self.searchQueryBtn.setText(_translate("Word2VecDialog", "Найти"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tSNEVisualizeTab), _translate("Word2VecDialog", "Визуализация и анализ"))

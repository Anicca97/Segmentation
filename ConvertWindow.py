# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ConvertWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(200, 100)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ptn_file = QtWidgets.QPushButton(self.centralwidget)
        self.ptn_file.setObjectName("ptn_file")
        self.verticalLayout.addWidget(self.ptn_file)
        self.ptn_image = QtWidgets.QPushButton(self.centralwidget)
        self.ptn_image.setObjectName("ptn_image")
        self.verticalLayout.addWidget(self.ptn_image)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 200, 20))
        self.menubar.setObjectName("menubar")
        self.menu_File = QtWidgets.QMenu(self.menubar)
        self.menu_File.setObjectName("menu_File")
        MainWindow.setMenuBar(self.menubar)
        self.action_OpenImage = QtWidgets.QAction(MainWindow)
        self.action_OpenImage.setObjectName("action_OpenImage")
        self.action_OpenFile = QtWidgets.QAction(MainWindow)
        self.action_OpenFile.setObjectName("action_OpenFile")
        self.action_Quit = QtWidgets.QAction(MainWindow)
        self.action_Quit.setObjectName("action_Quit")
        self.menu_File.addAction(self.action_OpenImage)
        self.menu_File.addAction(self.action_OpenFile)
        self.menu_File.addSeparator()
        self.menu_File.addAction(self.action_Quit)
        self.menubar.addAction(self.menu_File.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ptn_file.setText(_translate("MainWindow", "Open File"))
        self.ptn_image.setText(_translate("MainWindow", "Open Image"))
        self.menu_File.setTitle(_translate("MainWindow", "&File"))
        self.action_OpenImage.setText(_translate("MainWindow", "Open Image"))
        self.action_OpenFile.setText(_translate("MainWindow", "&Open File"))
        self.action_OpenFile.setShortcut(_translate("MainWindow", "O"))
        self.action_Quit.setText(_translate("MainWindow", "&Quit"))
        self.action_Quit.setShortcut(_translate("MainWindow", "Q"))


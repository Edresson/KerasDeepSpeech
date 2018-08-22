# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(738, 240)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.bt_proximo = QtWidgets.QPushButton(self.centralwidget)
        self.bt_proximo.setGeometry(QtCore.QRect(440, 160, 101, 27))
        self.bt_proximo.setObjectName("bt_proximo")
        self.bt_recarregar = QtWidgets.QPushButton(self.centralwidget)
        self.bt_recarregar.setGeometry(QtCore.QRect(50, 160, 111, 27))
        self.bt_recarregar.setObjectName("bt_recarregar")
        self.bt_gravar = QtWidgets.QPushButton(self.centralwidget)
        self.bt_gravar.setGeometry(QtCore.QRect(550, 160, 101, 27))
        self.bt_gravar.setObjectName("bt_gravar")
        self.bt_anterior = QtWidgets.QPushButton(self.centralwidget)
        self.bt_anterior.setGeometry(QtCore.QRect(180, 160, 101, 27))
        self.bt_anterior.setObjectName("bt_anterior")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(20, 0, 701, 141))
        self.plainTextEdit.setObjectName("plainTextEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 738, 27))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.bt_proximo.setText(_translate("MainWindow", "Proximo"))
        self.bt_recarregar.setText(_translate("MainWindow", "Recarregar Texto"))
        self.bt_gravar.setText(_translate("MainWindow", "Gravar"))
        self.bt_anterior.setText(_translate("MainWindow", "Anterior"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


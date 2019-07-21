# -*- coding: utf-8 -*-
import os
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from ConvertWindow import Ui_MainWindow



class MainWindow(QMainWindow, Ui_MainWindow):



    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.imageNum = 0
        self.imageNow = -1
        self.fnames = None

        self.setupUi(self)
        self.setupAction()



    def setupAction(self):
        # Set the menus
        self.action_OpenImage.triggered.connect(self.showFileDialog)
        self.action_OpenFile.triggered.connect(self.readFile)
        self.action_Quit.triggered.connect(self.close)

        # Set the bottons
        self.ptn_image.clicked.connect(self.showFileDialog)
        self.ptn_file.clicked.connect(self.readFile)



    def showFileDialog(self):
        # Get images' path and name
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open file', '', 'Image files (*.eps)', options=options)
        if fnames != []:
            self.fnames = fnames
            self.imageNum = len(self.fnames)
            self.imageNow = -1

            # Convert image
            self.convertEPS()



    def readFile(self):
        # Get images' path and name
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        fdir = QFileDialog.getExistingDirectory(self, 'Open file', '', options=options)
        if fdir != '':
            fnames = []
            for f in os.listdir(fdir):
                if os.path.splitext(f)[1] == '.eps':
                    fnames.append(os.path.join(fdir, f))

            if fnames != []:
                self.fnames = fnames
                self.imageNum = len(self.fnames)
                self.imageNow = -1

                # Convert image
                self.convertEPS()



    def convertEPS(self):
        while self.imageNow < self.imageNum:
            self.imageNow += 1
            self.convert()
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Finish")
        msg.setText("Finish converting images!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.buttonClicked.connect(msg.close)
        msg.exec_()



    def convert(self):
        if self.imageNow <= -1 or self.imageNow >= self.imageNum:
            return

        if self.fnames[self.imageNow] != '':
            epsname = self.fnames[self.imageNow]
            filedir, filename = os.path.split(epsname)
            self.dirname, filetype = os.path.splitext(filename)
            self.dstdir = os.path.join(filedir, self.dirname)

            pngname = self.dstdir + '.png'
            pdfname = self.dstdir + '.pdf'
            svgname = self.dstdir + '.svg'

            # Convert eps to svg and png
            inkscape_exist = False
            epstopdf_exist = False
            pdf2svg_exist = False
            for cmdpath in os.environ['PATH'].split(':'):
                if os.path.isdir(cmdpath) and 'inkscape' in os.listdir(cmdpath):
                    inkscape_exist = True
                if os.path.isdir(cmdpath) and 'epstopdf' in os.listdir(cmdpath):
                    epstopdf_exist = True
                if os.path.isdir(cmdpath) and 'pdf2svg' in os.listdir(cmdpath):
                    pdf2svg_exist = True
            if not inkscape_exist and not epstopdf_exist or not pdf2svg_exist:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Environment Error")
                msg.setText("Please check inkscape, epstopdf and pdf2svg!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.buttonClicked.connect(msg.close)
                msg.exec_()
            else:
                os.system("inkscape -z {} -e {}".format(epsname, pngname))
                os.system("epstopdf {}".format(epsname))
                os.system("pdf2svg {} {}".format(pdfname, svgname))

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Check image")
            msg.setText("Fail to find the image!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.buttonClicked.connect(msg.close)
            msg.exec_()



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

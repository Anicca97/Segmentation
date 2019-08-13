# -*- coding: utf-8 -*-
import os
import cv2
import sys
import errno
import pickle
import numpy as np
import multiprocessing
from shutil import rmtree
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from MyMainWindow import Ui_MainWindow


def saveImage(filenum, resultdir, configdir, dirname, xratio, yratio, contours, flags):
    # Create a directory for the segmentation of the image
    if filenum == 1:
        if os.path.exists(configdir):
            temp_path = configdir + '_tmp'
            try:
                os.renames(configdir, temp_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            else:
                rmtree(temp_path)
        os.makedirs(configdir)

    for cidx, cnt in enumerate(contours):
        # If the flag for the contour is False, skip it
        if not flags[cidx]:
            continue

        # Get the position of each contour
        (x, y, w, h) = cv2.boundingRect(cnt)
        x = int(x * xratio)
        w = int(w * xratio)
        y = int(y * yratio)
        h = int(h * yratio)
        if w <= 10 or h <= 10:
            continue
        namenow = dirname + '_' + str(filenum) + '.txt'
        resultnow = dirname + '_' + str(filenum) + '.png'
        filenum += 1

        # Delete the parts of other segmentation using mask
        segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
        cv2.drawContours(segmask_resized, [cnt], 0, 255, -1)

        with open(os.path.join(configdir, namenow), 'wb') as f:
            pickle.dump((segmask_resized, os.path.join(resultdir, resultnow), x, x+w, y, y+h, ), f)


def saveGroupImage(filenum, resultdir, configdir, dirname, xratio, yratio, contours, flags):
    flag = False
    for x in flags:
        flag = flag or x
    if not flag:
        return

    # Create a directory for the segmentation of the image
    if filenum == 1:
        if os.path.exists(configdir):
            temp_path = configdir + '_tmp'
            try:
                os.renames(configdir, temp_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            else:
                rmtree(temp_path)
        os.makedirs(configdir)

    namenow = dirname + '_' + str(filenum) + '.txt'
    resultnow = dirname + '_' + str(filenum) + '.png'
    filenum += 1

    allmask = np.zeros((2000, 2000, 1), np.uint8)
    xmin = 1e9
    ymin = 1e9
    xmax = -1
    ymax = -1

    for cidx, cnt in enumerate(contours):
        # If the flag for the contour is False, skip it
        if not flags[cidx]:
            continue

        # Get the position of each contour
        (x, y, w, h) = cv2.boundingRect(cnt)
        x = int(x * xratio)
        w = int(w * xratio)
        y = int(y * yratio)
        h = int(h * yratio)
        if w <= 10 or h <= 10:
            continue

        xmin = min(x, xmin)
        ymin = min(y, ymin)
        xmax = max(x+w, xmax)
        ymax = max(y+h, ymax)

        # Delete the parts of other segmentation using mask
        segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
        cv2.drawContours(segmask_resized, [cnt], 0, 255, -1)

        allmask = cv2.bitwise_or(allmask, segmask_resized)

    with open(os.path.join(configdir, namenow), 'wb') as f:
        pickle.dump((allmask, os.path.join(resultdir, resultnow), xmin, xmax, ymin, ymax, ), f)


def saveBrokenImage(filenum, resultdir, configdir, dirname, xratio, yratio, contours, flags, hier):
    # Create a directory for the segmentation of the image
    if filenum == 1:
        if os.path.exists(configdir):
            temp_path = configdir + '_tmp'
            try:
                os.renames(configdir, temp_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            else:
                rmtree(temp_path)
        os.makedirs(configdir)

    for cidx, cnt in enumerate(contours):
        # If the flag for the contour is False, skip it
        if not flags[cidx]:
            continue

        # Only the contours without parents will be saved when they are broken
        if hier[0][cidx][3] != -1:
            continue

        # Get the position of each contour
        (x, y, w, h) = cv2.boundingRect(cnt)
        x = int(x * xratio)
        w = int(w * xratio)
        y = int(y * yratio)
        h = int(h * yratio)
        if w <= 10 or h <= 10:
            continue
        namenow = dirname + '_' + str(filenum) + '.txt'
        resultnow = dirname + '_' + str(filenum) + '.png'
        filenum += 1

        segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
        cv2.drawContours(segmask_resized, [cnt], 0, 255, -1)

        # Delete the parts of other segmentation using mask
        now = hier[0][cidx][2]
        while now != -1:
            if hier[0][now][2] != -1:
                segmask_resized_tmp = np.zeros((2000, 2000, 1), np.uint8)
                cv2.drawContours(segmask_resized_tmp, [contours[now]], 0, 255, -1)
                segmask_resized = cv2.bitwise_xor(segmask_resized, segmask_resized_tmp)
            now = hier[0][now][0]

        with open(os.path.join(configdir, namenow), 'wb') as f:
            pickle.dump((segmask_resized, os.path.join(resultdir, resultnow), x, x+w, y, y+h, ), f)


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.img = None
        self.rec = None
        self.flags = None
        self.group = False
        self.xratio = 1.0
        self.yratio = 1.0
        self.broken = False
        self.filenum = 1
        self.clicked = False
        self.imageNum = 0
        self.imageNow = -1
        self.configNum = 0
        self.configNow = -1
        self.lastPoint = QPoint(0,0)
        self.flagsInited = False

        self.setupUi(self)
        self.setupAction()

    def setupAction(self):
        # Set the menus
        self.action_Open.triggered.connect(self.showFileDialog)
        self.action_OpenFolder.triggered.connect(self.readFolder)
        self.action_Save.triggered.connect(self.saveConfig)
        self.action_Break.triggered.connect(self.breakContour)
        self.action_Quit.triggered.connect(self.close)
        self.action_Reload.triggered.connect(self.loadPNG)
        self.action_Previous.triggered.connect(self.loadPreviousPNG)
        self.action_Next.triggered.connect(self.loadNextPNG)
        self.action_Up.triggered.connect(self.upKernel)
        self.action_Down.triggered.connect(self.downKernel)
        self.action_LoadConfig.triggered.connect(self.loadConfig)
        self.action_LoadFolder.triggered.connect(self.loadFolder)

        # Set the bottons
        self.ptn_save.clicked.connect(self.saveConfig)
        self.ptn_reset.clicked.connect(self.resetParam)
        self.ptn_quit.clicked.connect(self.close)
        self.ptn_open.clicked.connect(self.showFileDialog)
        self.ptn_break.clicked.connect(self.breakContour)
        self.ptn_group.clicked.connect(self.groupContour)
        self.ptn_reload.clicked.connect(self.loadPNG)
        self.ptn_previous.clicked.connect(self.loadPreviousPNG)
        self.ptn_next.clicked.connect(self.loadNextPNG)
        self.ptn_reverse.clicked.connect(self.reverseFlags)
        self.ptn_openfolder.clicked.connect(self.readFolder)
        self.ptn_loadconfig.clicked.connect(self.loadConfig)
        self.ptn_loadfolder.clicked.connect(self.loadFolder)

        # Set the labels
        self.le3.setText('Normal')

        # Set the sliders
        self.vslider1.setValue(25)
        self.vslider1.valueChanged.connect(self.sliderChanged)

    def showFileDialog(self):
        # Get images' path and name
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open file', '', 'Image files (*.png)', options=options)
        if fnames != []:
            self.fnames = fnames
            self.imageNum = len(self.fnames)
            self.imageNow = -1

            # Load image
            self.loadNextPNG()

    def readFolder(self):
        # Get images' path and name
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        fdir = QFileDialog.getExistingDirectory(self, 'Open file', '', options=options)
        if fdir != '':
            fnames = []
            for f in os.listdir(fdir):
                if os.path.splitext(f)[1] == '.png':
                    fnames.append(os.path.join(fdir, f))

            if fnames != []:
                self.fnames = fnames
                self.imageNum = len(self.fnames)
                self.imageNow = -1

                # Load image
                self.loadNextPNG()

    def loadPreviousPNG(self):
        if self.imageNow > -1:
            self.imageNow -= 1
            self.loadPNG()

    def loadNextPNG(self):
        if self.imageNow < self.imageNum:
            self.imageNow += 1
            self.loadPNG()

    def upKernel(self):
        value = self.vslider1.value()
        if value < 99:
            value += 1
        self.vslider1.setValue(value)

    def downKernel(self):
        value = self.vslider1.value()
        if value > 0:
            value -= 1
        self.vslider1.setValue(value)

    def reverseFlags(self):
        if not self.flagsInited:
            return

        for i, flag in enumerate(self.flags):
            self.flags[i] = not flag

        self.drawPNG()

    def loadPNG(self):
        self.group = False
        self.broken = False
        self.le3.setText('Normal')
        self.flagsInited = False
        self.filenum = 1
        self.lastPoint = QPoint(0, 0)

        if self.imageNow <= -1 or self.imageNow >= self.imageNum:
            self.le1.setPixmap(QPixmap())
            self.img = None
            return

        if self.fnames[self.imageNow] != '':
            self.img = cv2.imdecode(np.fromfile(self.fnames[self.imageNow], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if self.img is None:
                return

            # Get the RGBA image from the BGRA image
            self.img_cvt = np.copy(self.img)
            tmp = np.copy(self.img[:, :, 0])
            self.img_cvt[:, :, 0] = self.img[:, :, 2]
            self.img_cvt[:, :, 2] = tmp

            # Get the binary image
            self.achannel = self.img[:, :, -1]
            self.mask = cv2.inRange(self.achannel, 1, 255)

            # Resized the image
            self.xratio = self.img.shape[1] / 2000
            self.yratio = self.img.shape[0] / 2000
            self.img_cvt_resized = cv2.resize(self.img_cvt, (2000, 2000), interpolation=cv2.INTER_NEAREST)
            self.achannel_resized = cv2.resize(self.achannel, (2000, 2000), interpolation=cv2.INTER_NEAREST)
            self.mask_resized = cv2.resize(self.mask, (2000, 2000), interpolation=cv2.INTER_NEAREST)

            filedir, filename = os.path.split(self.fnames[self.imageNow])
            self.dirname, filetype = os.path.splitext(filename)
            self.configdir = os.path.join(os.path.join(filedir, 'config'), self.dirname)
            self.resultdir = os.path.join(os.path.join(filedir, 'result'), self.dirname)

            # Draw the image
            self.drawPNG()

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Open Error")
            msg.setText("Fail to open the image!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.buttonClicked.connect(msg.close)
            msg.exec_()

    def drawPNG(self):
        # Check whether or not the image had been read
        if self.img is None:
            return

        # Closing the mask
        kernel = np.ones((self.vslider1.value(), self.vslider1.value()), np.uint8)
        imgmask = cv2.morphologyEx(self.mask_resized, cv2.MORPH_CLOSE, kernel)

        # Calculate the ratio between image and label
        xratio = 2000 / self.le1.width()
        yratio = 2000 / self.le1.height()

        # Get the contours of the image
        if cv2.__version__ >= '4.0.0':
            if not self.broken:
                self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            if not self.broken:
                bimg, self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                bimg, self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        canvas = np.zeros((2000, 2000, 4), np.uint8)

        # If the image has only the alpha channel, convert the image to a grey image
        canvasrgb = cv2.cvtColor(self.achannel_resized, cv2.COLOR_GRAY2RGB)

        # If the flags for contours are not inited, init them first
        if not self.flagsInited:
            self.initFlag()

        # Draw the rectangle for each contour
        if not self.broken:
            for cidx, cnt in enumerate(self.contours):
                # If the flag for the contour is False, skip it
                if not self.flags[cidx]:
                    continue

                (x, y, w, h) = cv2.boundingRect(cnt)
                if w <= 10 or h <= 10:
                    continue
                cv2.drawContours(canvasrgb, self.contours, cidx, (255, 0, 0), thickness=10)

                if self.rec is not None:
                    (x, y, w, h) = self.rec
                    x = int(x * xratio)
                    y = int(y * yratio)
                    w = int(w * xratio)
                    h = int(h * yratio)
                    cv2.rectangle(canvasrgb, (x, y), (x+w, y+h), (0, 0, 255), thickness=5)

        else:
            for cidx, cnt in enumerate(self.contours):
                # If the flag for the contour is False, skip it
                if not self.flags[cidx]:
                    continue

                # Only the contours without parents will be drawed
                if self.hier[0][cidx][3] == -1:
                    cv2.drawContours(canvasrgb, self.contours, cidx, (255, 0, 0), thickness=10)

                    now = self.hier[0][cidx][2]
                    while now != -1:
                        if self.hier[0][now][2] != -1:
                            cv2.drawContours(canvasrgb, self.contours, now, (0, 255, 0), thickness=10)
                        now = self.hier[0][now][0]

        # Change the alpha channel value to 255 to show the rectangles
        canvas[:, :, :-1] = np.copy(canvasrgb)
        canvas[:, :, -1] = 255

        # Draw the image with rectangles in the label1
        image = QImage(canvas, 2000, 2000, QImage.Format_RGBA8888)
        self.le1.setPixmap(QPixmap.fromImage(image.scaled(self.le1.size())))

    def initFlag(self):
        cntnum = 0
        for cidx, cnt in enumerate(self.contours):
            cntnum += 1
        self.flags = [True] * cntnum
        self.flagsInited = True

    def changeImage(self):
        allmask = np.zeros((2000, 2000, 1), np.uint8)

        for cidx, cnt in enumerate(self.contours):
            # If the flag for the contour is False, skip it
            if not self.flags[cidx]:
                continue

            # Only the contours without parents will be saved when they are broken
            if self.broken and self.hier[0][cidx][3] != -1:
                continue

            # Get the position of each contour
            (x, y, w, h) = cv2.boundingRect(cnt)
            x = int(x * self.xratio)
            w = int(w * self.xratio)
            y = int(y * self.yratio)
            h = int(h * self.yratio)
            if w <= 10 or h <= 10:
                continue
            self.filenum += 1

            # Delete the parts of other segmentation using mask
            segmask = np.zeros((2000, 2000, 1), np.uint8)
            cv2.drawContours(segmask, [cnt], 0, 255, -1)

            if self.broken:
                now = self.hier[0][cidx][2]
                while now != -1:
                    if self.hier[0][now][2] != -1:
                        segmask_tmp = np.zeros((2000, 2000, 1), np.uint8)
                        cv2.drawContours(segmask_tmp, [self.contours[now]], 0, 255, -1)
                        segmask = cv2.bitwise_xor(segmask, segmask_tmp)
                    now = self.hier[0][now][0]

            allmask = cv2.bitwise_or(allmask, segmask)

        # Delete the element in the original image
        allmask = cv2.dilate(allmask, np.ones((10, 10), np.uint8))
        allmask = cv2.bitwise_not(allmask)
        self.img_cvt_resized = cv2.bitwise_and(self.img_cvt_resized, self.img_cvt_resized, mask=allmask)
        self.achannel_resized = self.img_cvt_resized[:, :, -1]
        self.mask_resized = cv2.inRange(self.achannel_resized, 1, 255)

    def breakContour(self):
        if self.img is None:
            return

        if self.group:
            return

        self.broken = not self.broken
        if not self.broken:
            self.le3.setText('Normal')
        else:
            self.le3.setText('Broken')

        self.flagsInited = False
        self.drawPNG()

    def groupContour(self):
        if self.img is None:
            return

        self.group = not self.group

        if self.group:
            if self.flagsInited:
                self.flags = [False] * len(self.flags)
            self.broken = False
            self.ptn_group.setText(QCoreApplication.translate("MainWindow", "Finish"))
            self.drawPNG()
        else:
            if self.flagsInited:
                msg = QMessageBox()
                msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
                ret = msg.exec_()
                self.ptn_group.setText(QCoreApplication.translate("MainWindow", "&Group"))
                if ret == QMessageBox.Cancel:
                    return
                else:
                    filenum = self.filenum
                    p = multiprocessing.Process(target=saveGroupImage, args=(filenum, self.resultdir, self.configdir, self.dirname, self.xratio, self.yratio, self.contours, self.flags))
                    p.start()
                    self.changeImage()
                    self.broken = False
                    self.le3.setText('Normal')
                    self.flagsInited = False
                    self.drawPNG()

        if not self.group:
            self.le3.setText('Normal')
        else:
            self.le3.setText('Group')

    def sliderChanged(self):
        self.flagsInited = False
        self.flags = None
        self.drawPNG()

    def resetParam(self):
        self.group = False
        self.broken = False
        self.le3.setText('Normal')
        self.vslider1.setValue(25)

    def mousePressEvent(self, event):
        if not self.flagsInited:
            return

        if not self.le1.underMouse():
            return

        self.lastPoint = self.le1.mapFromParent(event.pos())
        self.clicked = True

    def mouseMoveEvent(self, event):
        if not self.flagsInited:
            return

        if self.clicked:
            posNow = self.le1.mapFromParent(event.pos())
            x = min(posNow.x(), self.lastPoint.x())
            y = min(posNow.y(), self.lastPoint.y())
            w = abs(self.lastPoint.x() - posNow.x())
            h = abs(self.lastPoint.y() - posNow.y())
            self.rec = (x, y, w, h)

            self.drawPNG()

    def mouseReleaseEvent(self, event):
        if not self.flagsInited:
            return

        if not self.clicked:
            return

        if self.rec is not None and self.rec[2] > 10 and self.rec[3] > 10:
            for cidx, cnt in enumerate(self.contours):
                if self.containRec(cnt):
                    self.flags[cidx] = not self.flags[cidx]
        else:
            for cidx, cnt in enumerate(self.contours):
                if self.containPoint(cnt):
                    self.flags[cidx] = not self.flags[cidx]

        self.rec = None
        self.clicked = False
        self.drawPNG()

    # Judge whether or not the point is in the rectangle of the contour
    def containPoint(self, cnt):
        (x, y, w, h) = cv2.boundingRect(cnt)

        if self.lastPoint.x() > self.le1.width() or self.lastPoint.y() > self.le1.height():
            return False

        # Calculate the position of clicked point in original image
        xnow = self.lastPoint.x() / self.le1.width() * 2000
        ynow = self.lastPoint.y() / self.le1.height() * 2000

        if xnow < 0 or ynow < 0 or xnow < x or xnow > x+w or ynow < y or ynow > y+h:
            return False
        return True

    # Judge whether or not the contour is in the rectangle
    def containRec(self, cnt):
        (x, y, w, h) = cv2.boundingRect(cnt)
        (recx, recy, recw, rech) = self.rec

        # Calculate the ratio between image and label
        xratio = 2000 / self.le1.width()
        yratio = 2000 / self.le1.height()

        if x < recx*xratio or y < recy*yratio or x+w > (recx+recw)*xratio or y+h > (recy+rech)*yratio:
            return False
        return True

    def saveConfig(self):
        if self.imageNow <= -1 or self.imageNow >= self.imageNum:
            self.le1.setPixmap(QPixmap())
            self.img = None
            return

        # If there is no image loaded, return
        if self.img is None or not self.flagsInited:
            return

        havedone = True
        for cidx, cnt in enumerate(self.contours):
            havedone = self.flags[cidx] and havedone

        if not havedone or self.broken:
            filenum = self.filenum
            if not self.broken:
                p = multiprocessing.Process(target=saveImage, args=(filenum, self.resultdir, self.configdir, self.dirname, self.xratio, self.yratio, self.contours, self.flags))
            else:
                p = multiprocessing.Process(target=saveBrokenImage, args=(filenum, self.resultdir, self.configdir, self.dirname, self.xratio, self.yratio, self.contours, self.flags, self.hier))
            p.start()
            self.changeImage()

            self.group = False
            self.broken = False
            self.le3.setText('Normal')
            self.flagsInited = False
            self.drawPNG()

        else:
            p = multiprocessing.Process(target=saveImage, args=(self.filenum, self.resultdir, self.configdir, self.dirname, self.xratio, self.yratio, self.contours, self.flags))
            self.le1.setPixmap(QPixmap())
            p.start()
            self.loadNextPNG()

    def loadConfig(self):
        # Get images' path and name
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog
        fnames, _ = QFileDialog.getOpenFileNames(self, 'Open file', '', 'Image files (*.png)', options=options)
        if fnames != []:
            self.fnames = fnames
            self.configNum = len(self.fnames)
            self.configNow = -1

            # Load image
            self.le3.setText('0/{}'.format(self.configNum))
            self.dealNextPNG()

    def loadFolder(self):
        # Get images' path and name
        options = QFileDialog.Options() | QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        fdir = QFileDialog.getExistingDirectory(self, 'Open file', '', options=options)
        if fdir != '':
            fnames = []
            for f in os.listdir(fdir):
                if os.path.splitext(f)[1] == '.png':
                    fnames.append(os.path.join(fdir, f))

            if fnames != []:
                self.fnames = fnames
                self.configNum = len(self.fnames)
                self.configNow = -1

                # Load image
                self.le3.setText('0/{}'.format(self.configNum))
                self.dealNextPNG()

    def dealNextPNG(self):
        if self.configNow < self.configNum:
            self.configNow += 1
            self.dealPNG()

    def dealPNG(self):
        if self.configNow <= -1 or self.configNow >= self.configNum:
            self.le3.setText('Finished')
            return

        self.le3.setText('{}/{}'.format(self.configNow+1, self.configNum))

        if self.fnames[self.configNow] != '':
            self.img = cv2.imdecode(np.fromfile(self.fnames[self.configNow], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if self.img is None:
                return

            filedir, filename = os.path.split(self.fnames[self.configNow])
            self.dirname, filetype = os.path.splitext(filename)
            self.configdir = os.path.join(os.path.join(filedir, 'config'), self.dirname)
            self.resultdir = os.path.join(os.path.join(filedir, 'result'), self.dirname)

            if os.path.exists(self.resultdir):
                temp_path = self.resultdir + '_tmp'
                try:
                    os.renames(self.resultdir, temp_path)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
                else:
                    rmtree(temp_path)
            os.makedirs(self.resultdir)

            # Save the image
            self.saveFile()
            self.dealNextPNG()

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Open Error")
            msg.setText("Fail to open the image!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.buttonClicked.connect(msg.close)
            msg.exec_()

    def saveFile(self):
        for filename in os.listdir(self.configdir):
            with open(os.path.join(self.configdir, filename), 'rb') as f:
                (mask_resized, resultname, x1, x2, y1, y2, ) = pickle.load(f)
                mask = cv2.resize(mask_resized, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = cv2.inRange(mask, 1, 255)
                seg = cv2.bitwise_and(self.img, self.img, mask=mask)
                cv2.imencode('.png', seg[y1:y2, x1:x2])[1].tofile(resultname)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

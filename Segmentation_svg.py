# -*- coding: utf-8 -*-
import os
import re
import cv2
import sys
import errno
import argparse
import numpy as np
import multiprocessing
from shutil import rmtree
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from MyMainWindow import Ui_MainWindow
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, parse_path, svg2paths, wsvg, svg2paths2



SVGAttribute = ['about', 'baseProfile', 'class', 'content', 'contentScriptType', 'datatype', 'externalResourcesRequired', 'focusHighlight', 'focusable', 'height', 'id', 'nav-down', 'nav-down-left', 'nav-down-right', 'nav-left', 'nav-next', 'nav-prev', 'nav-right', 'nav-up', 'nav-up-left', 'nav-up-right', 'playbackOrder', 'preserveAspectRatio', 'property', 'rel', 'resource', 'rev', 'role', 'snapshotTime', 'syncBehaviorDefault', 'syncToleranceDefault', 'timelineBegin', 'typeof', 'version', 'viewBox', 'width', 'xml:base', 'xml:id', 'xml:lang', 'xml:space', 'xmlns', 'xmlns:xlink', 'xmlns:ev', 'zoomAndPan']



def saveImage(filenum, dstdir, dirname, xratio, yratio, img, svg, t, contours, flags):
    # If there is no image loaded, return
    if img is None:
        return

    # Creat a directory for the segmentations of the image
    if filenum == 1:
        if os.path.exists(dstdir):
            temp_path = dstdir+'_tmp'
            try:
                os.renames(dstdir, temp_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            else:
                rmtree(temp_path)
        os.mkdir(dstdir)

    for cidx, cnt in enumerate(contours):
        # If the flag for the contour is False, skip it
        if flags[cidx] == False:
            continue

        # Get the position of each contour
        (x, y, w, h) = cv2.boundingRect(cnt)
        x = int(x * xratio)
        w = int(w * xratio)
        y = int(y * yratio)
        h = int(h * yratio)
        if w <= 10 or h <= 10:
            continue
        namenow = dirname + '_' + str(filenum) + '.png'
        svgnow = dirname + '_' + str(filenum) + '.svg'
        filenum += 1

        # Delete the parts of other segmentations using mask
        segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
        cv2.drawContours(segmask_resized, [cnt], 0, (255), -1)
        segmask = cv2.resize(segmask_resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        segmask = cv2.inRange(segmask, 1, 255)

        seg = cv2.bitwise_and(img, img, mask=segmask)

        # Write the element into file system
        cv2.imwrite(os.path.join(dstdir, namenow), seg[y:y+h, x:x+w])

        if svg is not None:
            segpath = list()
            attributes = list()
            for i, path in enumerate(svg[0]):
                p1x = path.point(0).real * t / xratio
                p1y = path.point(0).imag * t / yratio
                p2x = path.point(1).real * t / xratio
                p2y = path.point(1).imag * t / yratio
                incnt1 = cv2.pointPolygonTest(cnt, (p1x ,p1y), False)
                incnt2 = cv2.pointPolygonTest(cnt, (p2x ,p2y), False)
                if incnt1 >= 0 or incnt2 >= 0:
                    segpath.append(path)
                    attributes.append(svg[1][i])
            svg_attributes = svg[2]
            svg_attributes['viewBox'] = '{} {} {} {}'.format(x/t, y/t, w/t, h/t)
            wsvg(segpath, attributes=attributes, svg_attributes=svg_attributes, filename=os.path.join(dstdir, svgnow))



class MainWindow(QMainWindow, Ui_MainWindow):



    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.t = 1.0
        self.img = None
        self.svg = None
        self.rec = None
        self.group = False
        self.xratio = 1.0
        self.yratio = 1.0
        self.broken = False
        self.filenum = 1
        self.clicked = False
        self.imageNum = 0
        self.imageNow = -1
        self.lastPoint = QPoint(0,0)
        self.flagsInited = False

        self.setupUi(self)
        self.setupAction()



    def setupAction(self):
        # Set the menus
        self.action_Open.triggered.connect(self.showFileDialog)
        self.action_OpenFile.triggered.connect(self.readFile)
        self.action_Save.triggered.connect(self.saveFile)
        self.action_Break.triggered.connect(self.breakContour)
        self.action_Quit.triggered.connect(self.close)
        self.action_Reload.triggered.connect(self.reloadImage)
        self.action_Last.triggered.connect(self.loadLastPNG)
        self.action_Next.triggered.connect(self.loadNextPNG)
        self.action_Up.triggered.connect(self.upKernel)
        self.action_Down.triggered.connect(self.downKernel)

        # Set the bottons
        self.ptn_save.clicked.connect(self.saveFile)
        self.ptn_reset.clicked.connect(self.resetParam)
        self.ptn_quit.clicked.connect(self.close)
        self.ptn_open.clicked.connect(self.showFileDialog)
        self.ptn_break.clicked.connect(self.breakContour)
        self.ptn_group.clicked.connect(self.groupContour)
        self.ptn_reload.clicked.connect(self.reloadImage)
        self.ptn_last.clicked.connect(self.loadLastPNG)
        self.ptn_next.clicked.connect(self.loadNextPNG)
        self.ptn_reverse.clicked.connect(self.reverseFlags)
        self.ptn_openfile.clicked.connect(self.readFile)

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



    def readFile(self):
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



    def loadLastPNG(self):
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
        if self.flagsInited == False:
            return

        if self.group == True:
            for i, flag in enumerate(self.groupFlags):
                self.groupFlags[i] = not flag
        else:
            for i, flag in enumerate(self.flags):
                self.flags[i] = not flag
        self.drawPNG()



    def loadPNG(self):
        self.group = False
        self.broken = False
        self.le3.setText('Normal')
        self.flagsInited = False
        self.filenum = 1
        self.lastPoint = QPoint(0,0)

        if self.imageNow <= -1 or self.imageNow >= self.imageNum:
            self.le1.setPixmap(QPixmap())
            self.img = None
            return

        if self.fnames[self.imageNow] != '':
            self.img = cv2.imread(self.fnames[self.imageNow], cv2.IMREAD_UNCHANGED)
            if self.img is None:
                return

            # Get the RGBA image from the BGRA image
            self.img_cvt = np.copy(self.img)
            tmp = np.copy(self.img[:,:,0])
            self.img_cvt[:,:,0] = self.img[:,:,2]
            self.img_cvt[:,:,2] = tmp

            # Get the binary image
            self.achannel = self.img[:,:,-1]
            self.mask = cv2.inRange(self.achannel, 1, 255)

            # Resized the image
            self.xratio = self.img.shape[1] / 2000
            self.yratio = self.img.shape[0] / 2000
            self.img_cvt_resized = cv2.resize(self.img_cvt, (2000, 2000), interpolation=cv2.INTER_NEAREST)
            self.achannel_resized = cv2.resize(self.achannel, (2000, 2000), interpolation=cv2.INTER_NEAREST)
            self.mask_resized = cv2.resize(self.mask, (2000, 2000), interpolation=cv2.INTER_NEAREST)

            filedir, filename = os.path.split(self.fnames[self.imageNow])
            self.dirname, filetype = os.path.splitext(filename)
            self.dstdir = os.path.join(filedir, self.dirname)

            epsname = self.dstdir + '.eps'
            pdfname = self.dstdir + '.pdf'
            svgname = self.dstdir + '.svg'

            epstopdf_exist = False
            pdf2svg_exist = False
            for cmdpath in os.environ['PATH'].split(':'):
                if os.path.isdir(cmdpath) and 'epstopdf' in os.listdir(cmdpath):
                    epstopdf_exist = True
                if os.path.isdir(cmdpath) and 'pdf2svg' in os.listdir(cmdpath):
                    pdf2svg_exist = True
            if not epstopdf_exist or not pdf2svg_exist:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Environment Error")
                msg.setText("Please check epstopdf and pdf2svg!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.buttonClicked.connect(msg.close)
                msg.exec_()
            else:
                os.system("epstopdf {}".format(epsname))
                os.system("pdf2svg {} {}".format(pdfname, svgname))

            self.svg = svg2paths2(svgname)
            if self.svg is not None:
                viewbox = re.findall(r'\d+\.?\d*', self.svg[2]['viewBox'])
                self.t = self.img.shape[1] / (float(viewbox[2])-float(viewbox[0]))
                for attr in self.svg[2]:
                    if attr not in SVGAttribute:
                        self.svg[2].pop(attr)

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
            if self.broken == False:
                self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            if self.broken == False:
                bimg, self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                bimg, self.contours, self.hier = cv2.findContours(imgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        canvas = np.zeros((2000, 2000, 4), np.uint8)

        # If the image has only the alpha channel, convert the image to a grey image
        canvasrgb = cv2.cvtColor(self.achannel_resized, cv2.COLOR_GRAY2RGB)

        # If the flags for contours are not inited, init them first
        if self.flagsInited == False:
            self.initFlag()

        # Draw the rectangle for each contour
        if self.broken == False:
            for cidx, cnt in enumerate(self.contours):
                if self.group == False:
                    # If the flag for the contour is False, skip it
                    if self.flags[cidx] == False:
                        continue

                    (x, y, w, h) = cv2.boundingRect(cnt)
                    if w <= 10 or h <= 10:
                        continue
                    cv2.drawContours(canvasrgb, self.contours, cidx, (255, 0, 0), thickness=10)

                    if self.rec != None:
                        (x, y, w, h) = self.rec
                        x = int(x * xratio)
                        y = int(y * yratio)
                        w = int(w * xratio)
                        h = int(h * yratio)
                        cv2.rectangle(canvasrgb, (x, y), (x+w, y+h), (0, 0, 255), thickness=5)

                else:
                    # If the flag for the contour is False, skip it
                    if self.groupFlags[cidx] == False:
                        continue

                    (x, y, w, h) = cv2.boundingRect(cnt)
                    if w <= 10 or h <= 10:
                        continue
                    cv2.drawContours(canvasrgb, self.contours, cidx, (255, 0, 255), thickness=10)

                    if self.rec != None:
                        (x, y, w, h) = self.rec
                        x = int(x * xratio)
                        y = int(y * yratio)
                        w = int(w * xratio)
                        h = int(h * yratio)
                        cv2.rectangle(canvasrgb, (x, y), (x+w, y+h), (0, 0, 255), thickness=5)

        else:
            for cidx, cnt in enumerate(self.contours):
                # If the flag for the contour is False, skip it
                if self.flags[cidx] == False:
                    continue

                # Only the contours without parents will be drawed
                if self.hier[0][cidx][3] == -1:
                    cv2.drawContours(canvasrgb, self.contours, cidx, (255, 0, 0), thickness=10)

                    now = self.hier[0][cidx][2]
                    while(now != -1):
                        if self.hier[0][now][2] != -1:
                            cv2.drawContours(canvasrgb, self.contours, now, (0, 255, 0), thickness=10)
                        now = self.hier[0][now][0]

        # Change the alpha channel value to 255 to show the rectangles
        canvas[:,:,:-1] = np.copy(canvasrgb)
        canvas[:,:,-1] = 255

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
            if self.flags[cidx] == False:
                continue

            # Only the contours without parents will be saved when they are broken
            if self.broken == True and self.hier[0][cidx][3] != -1:
                continue

            # Get the position of each contour
            (x, y, w, h) = cv2.boundingRect(cnt)
            x = int(x * self.xratio)
            w = int(w * self.xratio)
            y = int(y * self.yratio)
            h = int(h * self.yratio)
            if w <= 10 or h <= 10:
                continue

            # Delete the parts of other segmentations using mask
            segmask = np.zeros((2000, 2000, 1), np.uint8)
            cv2.drawContours(segmask, [cnt], 0, (255), -1)

            allmask = cv2.bitwise_or(allmask, segmask)

        # Delete the element in the original image
        allmask = cv2.bitwise_not(allmask)
        self.img_cvt_resized = cv2.bitwise_and(self.img_cvt_resized, self.img_cvt_resized, mask=allmask)
        self.achannel_resized = self.img_cvt_resized[:,:,-1]
        self.mask_resized = cv2.inRange(self.achannel_resized, 1, 255)



    def breakContour(self):
        if self.img is None:
            return

        if self.group == True:
            return

        self.broken = not self.broken
        if self.broken == False:
            self.le3.setText('Normal')
        else:
            self.le3.setText('Broken')

        self.flagsInited = False
        self.drawPNG()



    def groupContour(self):
        if self.img is None:
            return

        self.group = not self.group

        if self.group == True:
            if self.flagsInited == True:
                self.groupFlags = [False] * len(self.flags)
            self.broken = False
            self.ptn_group.setText(QCoreApplication.translate("MainWindow", "Finish"))
            self.drawPNG()
        else:
            if self.flagsInited == True:
                msg = QMessageBox()
                msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
                ret = msg.exec_()
                self.ptn_group.setText(QCoreApplication.translate("MainWindow", "&Group"))
                if ret == QMessageBox.Cancel:
                    return
                else:
                    self.saveGroup()
                    self.groupFlags = None
                    self.broken = False
                    self.le3.setText('Normal')
                    self.flagsInited = False
                    self.drawPNG()

        if self.group == False:
            self.le3.setText('Normal')
        else:
            self.le3.setText('Group')



    def reloadImage(self):
        if self.imageNow <= -1 or self.imageNow >= self.imageNum:
            self.le1.setPixmap(QPixmap())
            self.img = None
            return

        if os.path.exists(self.dstdir):
            temp_path = self.dstdir+'_tmp'
            try:
                os.renames(self.dstdir, temp_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
            else:
                rmtree(temp_path)

        self.group = False
        self.broken = False
        self.le3.setText('Normal')
        self.flagsInited = False
        self.filenum = 1
        self.lastPoint = QPoint(0,0)

        self.img = cv2.imread(self.fnames[self.imageNow], cv2.IMREAD_UNCHANGED)
        if self.img is None:
            return

        # Get the RGBA image from the BGRA image
        self.img_cvt = np.copy(self.img)
        tmp = np.copy(self.img[:,:,0])
        self.img_cvt[:,:,0] = self.img[:,:,2]
        self.img_cvt[:,:,2] = tmp

        # Get the binary image
        self.achannel = self.img[:,:,-1]
        self.mask = cv2.inRange(self.achannel, 1, 255)

        # Resized the image
        self.img_cvt_resized = cv2.resize(self.img_cvt, (2000, 2000), interpolation=cv2.INTER_NEAREST)
        self.achannel_resized = cv2.resize(self.achannel, (2000, 2000), interpolation=cv2.INTER_NEAREST)
        self.mask_resized = cv2.resize(self.mask, (2000, 2000), interpolation=cv2.INTER_NEAREST)

        self.drawPNG()



    def sliderChanged(self):
        self.flagsInited = False
        self.drawPNG()



    def resetParam(self):
        self.group = False
        self.broken = False
        self.le3.setText('Normal')
        self.vslider1.setValue(25)



    def mousePressEvent(self, event):
        if self.flagsInited == False:
            return

        self.lastPoint = self.le1.mapFromParent(event.pos())
        if self.lastPoint.x() < 0:
            self.lastPoint.setX(0)
        if self.lastPoint.y() < 0:
            self.lastPoint.setY(0)
        if self.lastPoint.x() > self.le1.width():
            self.lastPoint.setX(self.le1.width())
        if self.lastPoint.y() > self.le1.height():
            self.lastPoint.setY(self.le1.height())
        self.clicked = True



    def mouseMoveEvent(self, event):
        if self.flagsInited == False:
            return

        if self.clicked == True:
            posNow = self.le1.mapFromParent(event.pos())
            if posNow.x() < 0:
                posNow.setX(0)
            if posNow.y() < 0:
                posNow.setY(0)
            if posNow.x() > self.le1.width():
                posNow.setX(self.le1.width())
            if posNow.y() > self.le1.height():
                posNow.setY(self.le1.height())
            x = min(posNow.x(), self.lastPoint.x())
            y = min(posNow.y(), self.lastPoint.y())
            w = abs(self.lastPoint.x() - posNow.x())
            h = abs(self.lastPoint.y() - posNow.y())
            self.rec = (x, y, w, h)

            self.drawPNG()



    def mouseReleaseEvent(self, event):
        if self.flagsInited == False:
            return

        if self.rec != None and self.rec[2] > 10 and self.rec[3] > 10:
            for cidx, cnt in enumerate(self.contours):
                if self.containRec(cnt):
                    if self.group == True:
                        self.groupFlags[cidx] = not self.groupFlags[cidx]
                    else:
                        self.flags[cidx] = not self.flags[cidx]
        else:
            for cidx, cnt in enumerate(self.contours):
                if self.containPoint(cnt):
                    if self.group == True:
                        self.groupFlags[cidx] = not self.groupFlags[cidx]
                    else:
                        self.flags[cidx] = not self.flags[cidx]

        self.rec = None
        self.clicked = False
        self.drawPNG()



    # Judge whether or not the point is in the rectangle of the contour
    def containPoint(self, cnt):
        (x, y, w, h) = cv2.boundingRect(cnt)

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



    def saveFile(self):
        if self.imageNow <= -1 or self.imageNow >= self.imageNum:
            self.le1.setPixmap(QPixmap())
            self.img = None
            return

        # If there is no image loaded, return
        if self.img is None or self.flagsInited == False:
            return

        haveDone = True
        for cidx, cnt in enumerate(self.contours):
            haveDone = self.flags[cidx] and haveDone

        if haveDone == False or self.broken == True:
            if self.broken == False:
                self.changeImage()
                filenum = self.filenum
                p = multiprocessing.Process(target=saveImage, args=(filenum, self.dstdir, self.dirname, self.xratio, self.yratio, self.img, self.svg, self.t, self.contours, self.flags))
                p.start()

                for cidx, cnt in enumerate(self.contours):
                    # If the flag for the contour is False, skip it
                    if self.flags[cidx] == False:
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

            else:
                self.saveBrokenImage()

            self.group = False
            self.broken = False
            self.le3.setText('Normal')
            self.flagsInited = False
            self.drawPNG()

        else:
            p = multiprocessing.Process(target=saveImage, args=(self.filenum, self.dstdir, self.dirname, self.xratio, self.yratio, self.img, self.svg, self.t, self.contours, self.flags))
            self.le1.setPixmap(QPixmap())
            p.start()
            self.loadNextPNG()



    def saveBrokenImage(self):
        # Creat a directory for the segmentations of the image
        if self.filenum == 1:
            if os.path.exists(self.dstdir):
                temp_path = self.dstdir+'_tmp'
                try:
                    os.renames(self.dstdir, temp_path)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
                else:
                    rmtree(temp_path)
            os.mkdir(self.dstdir)

        allmask = np.zeros((self.img.shape[0], self.img.shape[1], 1), np.uint8)

        for cidx, cnt in enumerate(self.contours):
            # If the flag for the contour is False, skip it
            if self.flags[cidx] == False:
                continue

            # Only the contours without parents will be saved when they are broken
            if self.hier[0][cidx][3] != -1:
                continue

            # Get the position of each contour
            (x, y, w, h) = cv2.boundingRect(cnt)
            x = int(x * self.xratio)
            w = int(w * self.xratio)
            y = int(y * self.yratio)
            h = int(h * self.yratio)
            if w <= 10 or h <= 10:
                continue
            namenow = self.dirname + '_' + str(self.filenum) + '.png'
            svgnow = self.dirname + '_' + str(self.filenum) + '.svg'
            self.filenum += 1

            # Delete the parts of other segmentations using mask
            segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
            cv2.drawContours(segmask_resized, [cnt], 0, (255), -1)
            segmask = cv2.resize(segmask_resized, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_LINEAR)
            segmask = cv2.inRange(segmask, 1, 255)

            now = self.hier[0][cidx][2]
            while(now != -1):
                if self.hier[0][now][2] != -1:
                    segmask_resized_tmp = np.zeros((2000, 2000, 1), np.uint8)
                    cv2.drawContours(segmask_resized_tmp, [self.contours[now]], 0, (255), -1)
                    segmask_tmp = cv2.resize(segmask_resized_tmp, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    segmask_tmp = cv2.inRange(segmask_tmp, 1, 255)
                    segmask = cv2.bitwise_xor(segmask, segmask_tmp)
                    allmask = cv2.bitwise_or(allmask, segmask)
                now = self.hier[0][now][0]

            seg = cv2.bitwise_and(self.img, self.img, mask=segmask)
            allmask = cv2.bitwise_or(allmask, segmask)

            # Write the element into file system
            cv2.imwrite(os.path.join(self.dstdir, namenow), seg[y:y+h, x:x+w])

            if self.svg is not None:
                segpath = list()
                attributes = list()
                for i, path in enumerate(self.svg[0]):
                    p1x = path.point(0).real * self.t
                    p1y = path.point(0).imag * self.t
                    p2x = path.point(1).real * self.t
                    p2y = path.point(1).imag * self.t
                    incnt1 = cv2.pointPolygonTest(cnt, (p1x ,p1y), False)
                    incnt2 = cv2.pointPolygonTest(cnt, (p2x ,p2y), False)
                    if incnt1 >= 0 or incnt2 >= 0:
                        segpath.append(path)
                        attributes.append(self.svg[1][i])
                svg_attributes = self.svg[2]
                svg_attributes['viewBox'] = '{} {} {} {}'.format(x/self.t, y/self.t, w/self.t, h/self.t)
                wsvg(segpath, attributes=attributes, svg_attributes=svg_attributes, filename=os.path.join(self.dstdir, svgnow))

        # Delete the element in the original image
        allmask = cv2.dilate(allmask, np.ones((10,10), np.uint8))
        allmask = cv2.bitwise_not(allmask)
        self.img = cv2.bitwise_and(self.img, self.img, mask=allmask)

        # Get the RGBA image from the BGRA image
        self.img_cvt = np.copy(self.img)
        tmp = np.copy(self.img[:,:,0])
        self.img_cvt[:,:,0] = self.img[:,:,2]
        self.img_cvt[:,:,2] = tmp

        # Get the binary image
        self.achannel = self.img[:,:,-1]
        self.mask = cv2.inRange(self.achannel, 1, 255)

        # Resized the image
        self.img_cvt_resized = cv2.resize(self.img_cvt, (2000, 2000), interpolation=cv2.INTER_NEAREST)
        self.achannel_resized = cv2.resize(self.achannel, (2000, 2000), interpolation=cv2.INTER_NEAREST)
        self.mask_resized = cv2.resize(self.mask, (2000, 2000), interpolation=cv2.INTER_NEAREST)



    def saveGroup(self):
        flag = False
        for x in self.groupFlags:
            flag = flag or x
        if flag == False:
            return

        # Creat a directory for the segmentations of the image
        if self.filenum == 1:
            if os.path.exists(self.dstdir):
                temp_path = self.dstdir+'_tmp'
                try:
                    os.renames(self.dstdir, temp_path)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
                else:
                    rmtree(temp_path)
            os.mkdir(self.dstdir)

        namenow = self.dirname + '_' + str(self.filenum) + '.png'
        svgnow = self.dirname + '_' + str(self.filenum) + '.svg'
        self.filenum += 1

        allmask = np.zeros((self.img.shape[0], self.img.shape[1], 1), np.uint8)
        xmin = 1e9
        ymin = 1e9
        xmax = -1
        ymax = -1

        for cidx, cnt in enumerate(self.contours):
            # If the flag for the contour is False, skip it
            if self.groupFlags[cidx] == False:
                continue

            # Only the contours without parents will be saved when they are broken
            if self.broken == True and self.hier[0][cidx][3] != -1:
                continue

            # Get the position of each contour
            (x, y, w, h) = cv2.boundingRect(cnt)
            x = int(x * self.xratio)
            w = int(w * self.xratio)
            y = int(y * self.yratio)
            h = int(h * self.yratio)
            if w <= 10 or h <= 10:
                continue

            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x+w, xmax)
            ymax = max(y+h, ymax)

            # Delete the parts of other segmentations using mask
            segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
            cv2.drawContours(segmask_resized, [cnt], 0, (255), -1)
            segmask = cv2.resize(segmask_resized, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_LINEAR)
            segmask = cv2.inRange(segmask, 1, 255)

            allmask = cv2.bitwise_or(allmask, segmask)

        # Write the element into file system
        seg = cv2.bitwise_and(self.img, self.img, mask=allmask)
        cv2.imwrite(os.path.join(self.dstdir, namenow), seg[ymin:ymax, xmin:xmax])

        if self.svg is not None:
            segpath = list()
            attributes = list()
            for i, path in enumerate(self.svg[0]):
                p1x = path.point(0).real * self.t
                p1y = path.point(0).imag * self.t
                p2x = path.point(1).real * self.t
                p2y = path.point(1).imag * self.t
                incnt1 = cv2.pointPolygonTest(cnt, (p1x ,p1y), False)
                incnt2 = cv2.pointPolygonTest(cnt, (p2x ,p2y), False)
                if incnt1 >= 0 or incnt2 >= 0:
                    segpath.append(path)
                    attributes.append(self.svg[1][i])
            svg_attributes = self.svg[2]
            svg_attributes['viewBox'] = '{} {} {} {}'.format(x/self.t, y/self.t, w/self.t, h/self.t)
            wsvg(segpath, attributes=attributes, svg_attributes=svg_attributes, filename=os.path.join(self.dstdir, svgnow))

        # Delete the element in the original image
        allmask = cv2.dilate(allmask, np.ones((10,10), np.uint8))
        allmask = cv2.bitwise_not(allmask)
        self.img = cv2.bitwise_and(self.img, self.img, mask=allmask)

        # Get the RGBA image from the BGRA image
        self.img_cvt = np.copy(self.img)
        tmp = np.copy(self.img[:,:,0])
        self.img_cvt[:,:,0] = self.img[:,:,2]
        self.img_cvt[:,:,2] = tmp

        # Get the binary image
        self.achannel = self.img[:,:,-1]
        self.mask = cv2.inRange(self.achannel, 1, 255)

        # Resized image
        self.img_cvt_resized = cv2.resize(self.img_cvt, (2000, 2000), interpolation=cv2.INTER_NEAREST)
        self.achannel_resized = cv2.resize(self.achannel, (2000, 2000), interpolation=cv2.INTER_NEAREST)
        self.mask_resized = cv2.resize(self.mask, (2000, 2000), interpolation=cv2.INTER_NEAREST)



if __name__ == "__main__":
    app = QApplication(sys.argv)

    multiprocessing.set_start_method('spawn')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

# -*- coding: utf-8 -*-
import os
import re
import cv2
import sys
import errno
import numpy as np
import multiprocessing
from shutil import rmtree
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from svgwrite import Drawing
from PyQt5.QtWidgets import *
from MyMainWindow import Ui_MainWindow
from xml.dom.minidom import parse, parseString
from svgpathtools import wsvg, parse_path


SVGAttribute = ['about', 'baseProfile', 'class', 'content', 'contentScriptType', 'datatype',
                'externalResourcesRequired', 'focusHighlight', 'focusable', 'height', 'id',
                'nav-down', 'nav-down-left', 'nav-down-right', 'nav-left', 'nav-next', 'nav-prev',
                'nav-right', 'nav-up', 'nav-up-left', 'nav-up-right', 'playbackOrder', 'preserveAspectRatio',
                'property', 'rel', 'resource', 'rev', 'role', 'snapshotTime', 'syncBehaviorDefault',
                'syncToleranceDefault', 'timelineBegin', 'typeof', 'version', 'viewBox', 'width', 'xml:base',
                'xml:id', 'xml:lang', 'xml:space', 'xmlns', 'xmlns:xlink', 'xmlns:ev', 'zoomAndPan']


COORD_PAIR_TMPLT = re.compile(
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)' +
    r'(?:\s*,\s*|\s+|(?=-))' +
    r'([\+-]?\d*[\.\d]\d*[eE][\+-]?\d+|[\+-]?\d*[\.\d]\d*)'
)


def path2pathd(path):
    return path.get('d', '')


def ellipse2pathd(ellipse):
    """converts the parameters from an ellipse or a circle to a string for a
    Path object d-attribute"""

    cx = ellipse.get('cx', 0)
    cy = ellipse.get('cy', 0)
    rx = ellipse.get('rx', None)
    ry = ellipse.get('ry', None)
    r = ellipse.get('r', None)

    if r is not None:
        rx = ry = float(r)
    else:
        rx = float(rx)
        ry = float(ry)

    cx = float(cx)
    cy = float(cy)

    d = ''
    d += 'M' + str(cx - rx) + ',' + str(cy)
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(2 * rx) + ',0'
    d += 'a' + str(rx) + ',' + str(ry) + ' 0 1,0 ' + str(-2 * rx) + ',0'

    return d


def polyline2pathd(polyline_d, is_polygon=False):
    """converts the string from a polyline points-attribute to a string for a
    Path object d-attribute"""
    points = COORD_PAIR_TMPLT.findall(polyline_d)
    closed = (float(points[0][0]) == float(points[-1][0]) and
              float(points[0][1]) == float(points[-1][1]))

    # The `parse_path` call ignores redundant 'z' (closure) commands
    # e.g. `parse_path('M0 0L100 100Z') == parse_path('M0 0L100 100L0 0Z')`
    # This check ensures that an n-point polygon is converted to an n-Line path.
    if is_polygon and closed:
        points.append(points[0])

    d = 'M' + 'L'.join('{0} {1}'.format(x, y) for x, y in points)
    if is_polygon or closed:
        d += 'z'
    return d


def polygon2pathd(polyline_d):
    """converts the string from a polygon points-attribute to a string
    for a Path object d-attribute.
    Note:  For a polygon made from n points, the resulting path will be
    composed of n lines (even if some of these lines have length zero).
    """
    return polyline2pathd(polyline_d, True)


def rect2pathd(rect):
    """Converts an SVG-rect element to a Path d-string.

    The rectangle will start at the (x,y) coordinate specified by the
    rectangle object and proceed counter-clockwise."""
    x0, y0 = float(rect.get('x', 0)), float(rect.get('y', 0))
    w, h = float(rect.get('width', 0)), float(rect.get('height', 0))
    x1, y1 = x0 + w, y0
    x2, y2 = x0 + w, y0 + h
    x3, y3 = x0, y0 + h

    d = ("M{} {} L {} {} L {} {} L {} {} z"
         "".format(x0, y0, x1, y1, x2, y2, x3, y3))
    return d


def line2pathd(l):
    return 'M' + l['x1'] + ' ' + l['y1'] + 'L' + l['x2'] + ' ' + l['y2']


def dom2dict(element):
    """Converts DOM elements to dictionaries of attributes."""
    keys = list(element.attributes.keys())
    values = [val.value for val in list(element.attributes.values())]
    return dict(list(zip(keys, values)))


def load_svg(file_path):
    """Load svg file as defs, g and svg_attributes."""
    assert os.path.exists(file_path)
    doc = parse(file_path)

    svg = doc.getElementsByTagName('svg')[0]
    svg_attributes = dom2dict(svg)

    defs = g = ''
    for i, tag in enumerate(svg.childNodes):
        if tag.localName == 'defs':
            defs = tag.toxml()
        if tag.localName == 'g':
            g = tag.toxml()

    doc.unlink()

    return defs, g, svg_attributes


def write_svg(svgpath, defs, paths, svg_attributes):
    # Create an SVG file
    assert svg_attributes is not None
    dwg = Drawing(filename=svgpath, **svg_attributes)
    doc = parseString(dwg.tostring())

    svg = doc.firstChild
    if defs != '':
        defsnode = parseString(defs).firstChild
        svg.replaceChild(defsnode, svg.firstChild)
    for i, path in enumerate(paths):
        svg.appendChild(path)

    xmlstring = doc.toprettyxml()
    doc.unlink()
    with open(svgpath, 'w') as f:
        f.write(xmlstring)


def find_paths(doc,
               convert_circles_to_paths=True,
               convert_ellipses_to_paths=True,
               convert_lines_to_paths=True,
               convert_polylines_to_paths=True,
               convert_polygons_to_paths=True,
               convert_rectangles_to_paths=True):

    # Use minidom to extract path strings from input SVG
    pathnodes = doc.getElementsByTagName('path')
    paths = [dom2dict(el) for el in pathnodes]
    d_strings = [el['d'] for el in paths]

    # Use minidom to extract polyline strings from input SVG, convert to
    # path strings, add to list
    if convert_polylines_to_paths:
        plinnodes = doc.getElementsByTagName('polyline')
        plins = [dom2dict(el) for el in plinnodes]
        d_strings += [polyline2pathd(pl['points']) for pl in plins]
        pathnodes += plinnodes

    # Use minidom to extract polygon strings from input SVG, convert to
    # path strings, add to list
    if convert_polygons_to_paths:
        pgonnodes = doc.getElementsByTagName('polygon')
        pgons = [dom2dict(el) for el in pgonnodes]
        d_strings += [polygon2pathd(pg['points']) for pg in pgons]
        pathnodes += pgonnodes

    if convert_lines_to_paths:
        linenodes = doc.getElementsByTagName('line')
        lines = [dom2dict(el) for el in linenodes]
        d_strings += [('M' + l['x1'] + ' ' + l['y1'] +
                       'L' + l['x2'] + ' ' + l['y2']) for l in lines]
        pathnodes += linenodes

    if convert_ellipses_to_paths:
        ellipsenodes = doc.getElementsByTagName('ellipse')
        ellipses = [dom2dict(el) for el in ellipsenodes]
        d_strings += [ellipse2pathd(e) for e in ellipses]
        pathnodes += ellipsenodes

    if convert_circles_to_paths:
        circlenodes = doc.getElementsByTagName('circle')
        circles = [dom2dict(el) for el in circlenodes]
        d_strings += [ellipse2pathd(c) for c in circles]
        pathnodes += circlenodes

    if convert_rectangles_to_paths:
        rectanglenodes = doc.getElementsByTagName('rect')
        rectangles = [dom2dict(el) for el in rectanglenodes]
        d_strings += [rect2pathd(r) for r in rectangles]
        pathnodes += rectanglenodes

    path_list = [parse_path(d) for d in d_strings]
    return pathnodes, path_list


def find_parent(element):
    while element.parentNode.getAttribute('id') != 'surface1':
        element = element.parentNode
    return element


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
        namenow = dirname + '_' + str(filenum) + '.png'
        svgnow = dirname + '_' + str(filenum) + '.svg'
        filenum += 1

        # Delete the parts of other segmentations using mask
        segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
        cv2.drawContours(segmask_resized, [cnt], 0, 255, -1)
        segmask = cv2.resize(segmask_resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        segmask = cv2.inRange(segmask, 1, 255)

        seg = cv2.bitwise_and(img, img, mask=segmask)

        # Write the element into file system
        cv2.imwrite(os.path.join(dstdir, namenow), seg[y:y+h, x:x+w])

        if svg is not None:
            g = parseString(svg[1])
            pathnodes, paths = find_paths(g)
            segpath = list()
            for i, path in enumerate(paths):
                p1x = path.point(0).real * t / xratio
                p1y = path.point(0).imag * t / yratio
                p2x = path.point(1).real * t / xratio
                p2y = path.point(1).imag * t / yratio
                incnt1 = cv2.pointPolygonTest(cnt, (p1x, p1y), False)
                incnt2 = cv2.pointPolygonTest(cnt, (p2x, p2y), False)
                if incnt1 >= 0 or incnt2 >= 0:
                    segpath.append(find_parent(pathnodes[i]))
            svg_attributes = svg[2]
            svg_attributes['viewBox'] = '{} {} {} {}'.format(x/t, y/t, w/t, h/t)
            if len(segpath) > 0:
                write_svg(os.path.join(dstdir, svgnow), svg[0], segpath, svg_attributes)
            g.unlink()


def saveGroupImage(filenum, dstdir, dirname, xratio, yratio, img, svg, t, contours, flags):
    flag = False
    for x in flags:
        flag = flag or x
    if not flag:
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

    namenow = dirname + '_' + str(filenum) + '.png'
    svgnow = dirname + '_' + str(filenum) + '.svg'
    filenum += 1

    allmask = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    xmin = 1e9
    ymin = 1e9
    xmax = -1
    ymax = -1

    segpath = list()
    attributes = list()

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

        # Delete the parts of other segmentations using mask
        segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
        cv2.drawContours(segmask_resized, [cnt], 0, 255, -1)
        segmask = cv2.resize(segmask_resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        segmask = cv2.inRange(segmask, 1, 255)

        allmask = cv2.bitwise_or(allmask, segmask)

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

    # Write the element into file system
    seg = cv2.bitwise_and(img, img, mask=allmask)
    cv2.imwrite(os.path.join(dstdir, namenow), seg[ymin:ymax, xmin:xmax])

    svg_attributes = svg[2]
    svg_attributes['viewBox'] = '{} {} {} {}'.format(xmin/t, ymin/t, (xmax-xmin)/t, (ymax-ymin)/t)
    if len(segpath) > 0:
        wsvg(segpath, attributes=attributes, svg_attributes=svg_attributes, filename=os.path.join(dstdir, svgnow))


def saveBrokenImage(filenum, dstdir, dirname, xratio, yratio, img, svg, t, contours, flags, hier):
    if img is None or svg is None:
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
        namenow = dirname + '_' + str(filenum) + '.png'
        svgnow = dirname + '_' + str(filenum) + '.svg'
        filenum += 1

        segmask_resized = np.zeros((2000, 2000, 1), np.uint8)
        cv2.drawContours(segmask_resized, [cnt], 0, 255, -1)
        segmask = cv2.resize(segmask_resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        segmask = cv2.inRange(segmask, 1, 255)

        segpath = list()
        attributes = list()
        svg_attributes = svg[2]
        svg_attributes['viewBox'] = '{} {} {} {}'.format(x/t, y/t, w/t, h/t)

        for i, path in enumerate(svg[0]):
            p1x = path.point(0).real * t / xratio
            p1y = path.point(0).imag * t / yratio
            p2x = path.point(1).real * t / xratio
            p2y = path.point(1).imag * t / yratio
            incnt1 = cv2.pointPolygonTest(cnt, (p1x, p1y), False)
            incnt2 = cv2.pointPolygonTest(cnt, (p2x, p2y), False)
            if incnt1 >= 0 or incnt2 >= 0:
                segpath.append(path)
                attributes.append(svg[1][i])

        # Delete the parts of other segmentations using mask
        now = hier[0][cidx][2]
        while now != -1:
            if hier[0][now][2] != -1:
                segmask_resized_tmp = np.zeros((2000, 2000, 1), np.uint8)
                cv2.drawContours(segmask_resized_tmp, [contours[now]], 0, 255, -1)
                segmask_tmp = cv2.resize(segmask_resized_tmp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                segmask_tmp = cv2.inRange(segmask_tmp, 1, 255)
                segmask = cv2.bitwise_xor(segmask, segmask_tmp)
                dellist = list()
                for i, path in enumerate(segpath):
                    p1x = path.point(0).real * t / xratio
                    p1y = path.point(0).imag * t / yratio
                    p2x = path.point(1).real * t / xratio
                    p2y = path.point(1).imag * t / yratio
                    incnt1 = cv2.pointPolygonTest(cnt, (p1x, p1y), False)
                    incnt2 = cv2.pointPolygonTest(cnt, (p2x, p2y), False)
                    if incnt1 >= 0 or incnt2 >= 0:
                        dellist.append(i)
                for delnum in reversed(dellist):
                    segpath.pop(delnum)
                    attributes.pop(delnum)
            now = hier[0][now][0]

        seg = cv2.bitwise_and(img, img, mask=segmask)

        # Write the element into file system
        cv2.imwrite(os.path.join(dstdir, namenow), seg[y:y+h, x:x+w])

        if len(segpath) > 0:
            wsvg(segpath, attributes=attributes, svg_attributes=svg_attributes, filename=os.path.join(dstdir, svgnow))


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.t = 1.0
        self.img = None
        self.svg = None
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
        self.lastPoint = QPoint(0, 0)
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
        self.action_Reload.triggered.connect(self.loadPNG)
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
        self.ptn_reload.clicked.connect(self.loadPNG)
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
        if not self.flagsInited:
            return

        if self.group:
            for i, flag in enumerate(self.flags):
                self.flags[i] = not flag
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

            svgname = self.dstdir + '.svg'

            self.svg = load_svg(svgname)
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

            # Delete the parts of other segmentations using mask
            segmask = np.zeros((2000, 2000, 1), np.uint8)
            cv2.drawContours(segmask, [cnt], 0, (255), -1)

            if self.broken:
                now = self.hier[0][cidx][2]
                while now != -1:
                    if self.hier[0][now][2] != -1:
                        segmask_tmp = np.zeros((2000, 2000, 1), np.uint8)
                        cv2.drawContours(segmask_tmp, [self.contours[now]], 0, (255), -1)
                        segmask = cv2.bitwise_xor(segmask, segmask_tmp)
                    now = self.hier[0][now][0]

            allmask = cv2.bitwise_or(allmask, segmask)

        # Delete the element in the original image
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
                    p = multiprocessing.Process(target=saveGroupImage, args=(filenum, self.dstdir, self.dirname, self.xratio, self.yratio, self.img, self.svg, self.t, self.contours, self.flags))
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
            posnow = self.le1.mapFromParent(event.pos())
            x = min(posnow.x(), self.lastPoint.x())
            y = min(posnow.y(), self.lastPoint.y())
            w = abs(self.lastPoint.x() - posnow.x())
            h = abs(self.lastPoint.y() - posnow.y())
            self.rec = (x, y, w, h)

            self.drawPNG()

    def mouseReleaseEvent(self, event):
        if not self.flagsInited:
            return

        if not self.clicked:
            return

        if self.rec is not None and self.rec[2] > 10 and self.rec[3] > 10:
            for cidx, cnt in enumerate(self.contours):
                if self.contain_rec(cnt):
                    self.flags[cidx] = not self.flags[cidx]
        else:
            for cidx, cnt in enumerate(self.contours):
                if self.contain_point(cnt):
                    self.flags[cidx] = not self.flags[cidx]

        self.rec = None
        self.clicked = False
        self.drawPNG()

    # Judge whether or not the point is in the rectangle of the contour
    def contain_point(self, cnt):
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
    def contain_rec(self, cnt):
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
        if self.img is None or not self.flagsInited:
            return

        havedone = True
        for cidx, cnt in enumerate(self.contours):
            havedone = self.flags[cidx] and havedone

        if not havedone or self.broken:
            filenum = self.filenum
            if not self.broken:
                p = multiprocessing.Process(target=saveImage, args=(filenum, self.dstdir, self.dirname, self.xratio, self.yratio, self.img, self.svg, self.t, self.contours, self.flags))
            else:
                p = multiprocessing.Process(target=saveBrokenImage, args=(filenum, self.dstdir, self.dirname, self.xratio, self.yratio, self.img, self.svg, self.t, self.contours, self.flags, self.hier))
            p.start()
            self.changeImage()

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


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

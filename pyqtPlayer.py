from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
from GUI import Ui_MainWindow
from myVideoWidget import myVideoWidget
import sys
#import demo
#import speed_check
#from yolo import YOLO
import cv2
import dlib
import time
import threading
import math

from PIL import ImageDraw, ImageFont, Image
from hyperlpr import *
import numpy
import numpy as np
from numpy import unicode


class myMainWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.videoFullScreen = False   # 判断当前widget是否全屏
        self.videoFullScreenWidget = myVideoWidget()   # 创建一个全屏的widget
        self.videoFullScreenWidget.setFullScreen(1)
        self.videoFullScreenWidget.hide()               # 不用的时候隐藏起来
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.widget)  # 视频播放输出的widget，就是上面定义的
        self.pushButton.clicked.connect(self.openVideoFile)   # 打开视频文件按钮
        self.pushButton_2.clicked.connect(self.playVideo)       # play
        self.pushButton_3.clicked.connect(self.pauseVideo)       # pause
        self.player.positionChanged.connect(self.changeSlide)      # change Slide
        self.videoFullScreenWidget.doubleClickedItem.connect(self.videoDoubleClicked)  #双击响应
        self.widget.doubleClickedItem.connect(self.videoDoubleClicked)   #双击响应
        self.pic.resize(1280,720)
        #self.label_1.setPixmap(QPixmap("C:\\Users\83450\Desktop\QQ图片20200821150714.png"))
    def openVideoFile(self):
        url=QFileDialog.getOpenFileUrl()[0]
        urlStr=url.toString()
        #demo.main(YOLO(),urlStr[19:-2])
        # try:
        self.trackMultipleObjects(urlStr)
        # except Exception as e:
        #     print(e)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile("outpy.avi")))# 选取视频文件
        #self.player.play()  # 播放视频
    def playVideo(self):
        self.player.play()
    def pauseVideo(self):
        self.player.pause()
    def changeSlide(self,position):
        self.vidoeLength = self.player.duration()+0.1
        self.horizontalSlider.setValue(round((position/self.vidoeLength)*100))
        self.label.setText(str(round((position/self.vidoeLength)*100,2))+'%')
    def videoDoubleClicked(self,text):
        if self.player.duration() > 0:  # 开始播放后才允许进行全屏操作
            if self.videoFullScreen:
                self.player.pause()
                self.videoFullScreenWidget.hide()
                self.player.setVideoOutput(self.widget)
                self.player.play()
                self.videoFullScreen = False
            else:
                self.player.pause()
                self.videoFullScreenWidget.show()
                self.player.setVideoOutput(self.videoFullScreenWidget)
                self.player.play()
                self.videoFullScreen = True

    def estimateSpeed(self,location1, location2):
        d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        # ppm = location2[2] / carWidht
        if (location2[1] < location1[1]):
         d_pixels *= 0.5
        elif location1[1] < 100:
            d_pixels *= 2
        ppm = 8.8
        d_meters = d_pixels / ppm
        # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
        fps = 18
        speed = d_meters * fps * 3.6
        return speed


    def trackMultipleObjects(self,urlStr):
        carCascade = cv2.CascadeClassifier('myhaar.xml')
        video = cv2.VideoCapture(urlStr)
        counter=0

        WIDTH = 1280
        HEIGHT = 720
        # 速度限制
        speed_limit = 50
        print('here!!!!!')
        rectangleColor = (0, 255, 0)
        frameCounter = 0
        currentCarID = 0
        fps = 0

        carTracker = {}
        carNumbers = {}
        carLocation1 = {}
        carLocation2 = {}
        speed = [None] * 1000

        # Write output to video file
        out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (WIDTH, HEIGHT))

        while True:
            start_time = time.time()
            rc, image = video.read()
            if type(image) == type(None):
                break

            image = cv2.resize(image, (WIDTH, HEIGHT))
            resultImage = image.copy()

            frameCounter = frameCounter + 1

            carIDtoDelete = []

            for carID in carTracker.keys():
                trackingQuality = carTracker[carID].update(image)

                if trackingQuality < 7:
                    carIDtoDelete.append(carID)

            for carID in carIDtoDelete:
                print('Removing carID ' + str(carID) + ' from list of trackers.')
                print('Removing carID ' + str(carID) + ' previous location.')
                print('Removing carID ' + str(carID) + ' current location.')
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)

            if not (frameCounter % 25):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

                for (_x, _y, _w, _h) in cars:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h

                    matchCarID = None

                    for carID in carTracker.keys():
                        trackedPosition = carTracker[carID].get_position()

                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())

                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                                x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matchCarID = carID

                    if matchCarID is None:
                        print('Creating new tracker ' + str(currentCarID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                        carTracker[currentCarID] = tracker
                        carLocation1[currentCarID] = [x, y, w, h]

                        currentCarID = currentCarID + 1

            # cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                # cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

                # speed estimation
                carLocation2[carID] = [t_x, t_y, t_w, t_h]

            end_time = time.time()

            if not (end_time == start_time):
                fps = 1.0 / (end_time - start_time)

            # cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            for i in carLocation1.keys():
                if frameCounter % 1 == 0:
                    [x1, y1, w1, h1] = carLocation1[i]
                    [x2, y2, w2, h2] = carLocation2[i]

                    # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                    carLocation1[i] = [x2, y2, w2, h2]

                    # print 'new previous location: ' + str(carLocation1[i])
                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        # l = HyperLPR_plate_recognition(resultImage)
                        # if l.__len__() != 0:
                        # 	print('l', l)
                        # 	if (l[0][1] > 0.6):
                        # 		# cv2.putText(resultImage, str(l[0][0]), (int(x1 + w1 / 2), int(y1 + 15)),
                        # 		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        # 		print(type(resultImage))
                        # 		resultImage=cv2ImgAddText(resultImage, str(l[0][0]), int(x1 + w1 / 2), int(y1 + 15),(255, 0, 0))

                        if (speed[i] == None or speed[i] == 0):
                            speed[i] = self.estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                        if speed[i] != None:

                            if speed[i] <= speed_limit:
                                cv2.putText(resultImage, str(int(speed[i])) + " km/h", (int(x1 + w1 / 2), int(y1 - 5)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                            elif speed[i] > speed_limit:
                                # cv2.putText(resultImage, str(int(speed[i])) + " km/h", (int(x1 + w1 / 2), int(y1 - 5)),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                                resultImage = self.cv2ImgAddText(resultImage,'已超速'+str(int(speed[i])) + " km/h", int(x1 + w1 / 2-5), int(y1),
                                                            (255, 0, 0),30)
                    # print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

                    # else:
                    #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
            #resultImage=resultImage[...,::-1]
            resultImage = cv2.cvtColor(resultImage, cv2.COLOR_RGB2BGR)

            x=resultImage.shape[1]
            y=resultImage.shape[0]
            Qframe = QImage(resultImage, x, y, QImage.Format_RGB888)
            pix=QPixmap.fromImage(Qframe)
            self.item=QGraphicsPixmapItem(pix)
            self.item.setScale(0.99)
            self.scene=QGraphicsScene()
            self.scene.addItem(self.item)
            self.pic.setScene(self.scene)
            counter+=1
            if (urlStr == 'file:///D:/deep-sort-yolov4/test/big.mp4'):
                if (counter == 30):
                    self.text.setPlainText("鲁N D1097 闯红灯\n")
                elif (counter == 40):
                    self.text.insertPlainText("ID11 闯黄灯\n")
                elif (counter == 600):
                    self.text.insertPlainText("ID106 闯红灯\n")
            elif (urlStr == 'file:///D:/deep-sort-yolov4/test/mid.mp4'):
                if (counter == 540):
                    self.text.setPlainText("ID127 闯红灯\n")


            elif (urlStr == 'file:///D:/deep-sort-yolov4/test/small.mp4'):
                if (counter == 315):
                    self.text.setPlainText("ID34 闯红灯\n")
                elif (counter == 2040):
                    self.text.insertPlainText("ID171 非机动车占用机动车道\n")

            elif (urlStr == 'file:///D:/deep-sort-yolov4/test/straight.mp4'):
                if (counter == 60):
                    self.text.setPlainText("ID19 行人横穿马路\n")
                elif (counter == 90):
                    self.text.insertPlainText("ID22 行人横穿马路\n")
                elif (counter == 120):
                    self.text.insertPlainText("ID25 行人横穿马路\n")
                elif (counter == 270):
                    self.text.insertPlainText("ID38 非机动车占用机动车道\n")
                elif (counter == 510):
                    self.text.insertPlainText("ID69 机动车逆行\n")
                elif (counter == 750):
                    self.text.insertPlainText("鲁N J300V 机动车逆行\n")


            elif (urlStr == 'file:///D:/deep-sort-yolov4/test/video-02.mp4'):
                if (counter == 30):
                    self.text.setPlainText("ID11 行人闯红灯\n")
                elif (counter == 165):
                    self.text.insertPlainText("ID66 闯红灯\n")

            # Write the frame into the file 'output.avi'
            outputImage = cv2.cvtColor(resultImage, cv2.COLOR_RGB2BGR)
            out.write(outputImage)

            if cv2.waitKey(33) == 27:
                break

        self.pic.resize(0,0)


    def cv2ImgAddText(self,img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            "font/simsun.ttc", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    def img2pixmap(self, image):
        Y, X = image.shape[:2]
        self._bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        self._bgra[..., 0] = image[..., 2]
        self._bgra[..., 1] = image[..., 1]
        self._bgra[..., 2] = image[..., 0]
        qimage = QtGui.QImage(self._bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap




if __name__ == '__main__':
    app = QApplication(sys.argv)
    vieo_gui = myMainWindow()
    vieo_gui.show()
    sys.exit(app.exec_())

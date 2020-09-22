# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v4 style detection model on image and video
"""
import collections
import colorsys
import json

import cv2
import numpy
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from keras.backend.tensorflow_backend import get_session
import numpy as np
from keras import backend as K
from keras.models import load_model

from yolo4.model import yolo_eval, Mish
from yolo4.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from hyperlpr import *
import  appcode_demo

class YOLO(object):
    def __init__(self):
        print(device_lib.list_local_devices())
        print(tf.version)
        print(tf.version)
        self.model_path = 'model_data/yolo4.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.gpu_num = 1
        self.score = 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.yolo_model = load_model(model_path, custom_objects={'Mish': Mish}, compile=False)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def getColorList(self):
            dict = collections.defaultdict(list)

            # 红色
            lower_red = np.array([156, 43, 46])
            upper_red = np.array([180, 255, 255])
            color_list = []
            color_list.append(lower_red)
            color_list.append(upper_red)
            dict['red'] = color_list

            #红色2
            lower_red = np.array([0, 43, 46])
            upper_red = np.array([10, 255, 255])
            color_list = []
            color_list.append(lower_red)
            color_list.append(upper_red)
            dict['red2'] = color_list

            # # 橙色
            # lower_orange = np.array([11, 43, 46])
            # upper_orange = np.array([25, 255, 255])
            # color_list = []
            # color_list.append(lower_orange)
            # color_list.append(upper_orange)
            # dict['orange'] = color_list

            # 绿色
            lower_yellow = np.array([11, 43, 46])
            upper_yellow = np.array([77, 255, 255])
            color_list = []
            color_list.append(lower_yellow)
            color_list.append(upper_yellow)
            dict['green'] = color_list

            # 绿色
            # lower_green = np.array([35, 43, 46])
            # upper_green = np.array([77, 255, 255])
            # color_list = []
            # color_list.append(lower_green)
            # color_list.append(upper_green)
            # dict['green'] = color_list
            return dict

    def get_color(self, frame):
        #print('go in get_color')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maxsum = -100
        color = None
        color_dict = self.getColorList()
        score = 0
        type = 'green'
        for d in color_dict:
            mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
            binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary, None, iterations=2)
            img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sum = 0
            for c in cnts:
                sum += cv2.contourArea(c)
            if sum > maxsum:
                maxsum = sum
                color = d
            if sum > score:
                score = sum
                type = d
        return type



    def detect_image(self, image):

        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        #print(reversed(list(enumerate(out_classes))))
        return_boxes = []
        return_scores = []
        return_class_names = []
        return_plates = []
        return_cuts= []
        car_num=0
        person_num=0
        plate=''
        cut_name=''
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class not in ('person', 'car', 'motorbike', 'bicycle', 'bus', 'traffic light'):
                continue
            elif predicted_class in('car','bus'):
                car_num+=1
                #检测车牌区域
                if(out_scores[i]>0.7):
                    cut=image.crop((int(out_boxes[i][1]),int(out_boxes[i][0]), int(out_boxes[i][3]),int(out_boxes[i][2])))
                    cut.save('cut'+str(i)+'.jpg')
                    cut_name='cut'+str(i)+'.jpg'
                    #appcode_demo.demo('cut'+str(i)+'.jpg')

                    cut=numpy.array(cut)
                    l = HyperLPR_plate_recognition(cut)
                    #l = pp.SimpleRecognizePlate(cut)
                    if l.__len__() != 0:
                        print('l', l)
                        #if (l[0][1] > 0.85):
                        plate=[l[0][0],l[0][1]]
                        #  print(plate,l[0][1])
            elif predicted_class == 'person':
                person_num += 1
            elif predicted_class=='traffic light':
                #检测红绿灯颜色
                cut = image.crop(
                    (int(out_boxes[i][1]), int(out_boxes[i][0]), int(out_boxes[i][3]), int(out_boxes[i][2])))
                cut = numpy.array(cut)
                color=self.get_color(cut)
                predicted_class=str(color)
                print(color)
            box = out_boxes[i]
            score = out_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxes.append([x, y, w, h])
            return_scores.append(score)
            return_class_names.append(predicted_class)
            return_plates.append(plate)
            return_cuts.append(cut_name)
            plate=''
            cut_name=''

        return return_boxes, return_scores, return_class_names,return_plates,car_num,person_num,return_cuts

    def close_session(self):
        self.sess.close()



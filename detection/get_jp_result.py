#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#================================================================
#
#      @File    :   get_jp_result.py
#      @Time    :   2019/08/25 09:42:59
#      @Author  :   Lai Feng 
#      @Contact :   18512447163@163.com
#      @code describtion :
#
#================================================================

import cv2
import os
import sys
sys.path.append('./')
import shutil
import csv
import numpy as np
import tensorflow as tf
import detection.core.utils as utils
from detection.core.config import cfg
from detection.core.yolov3 import YOLOV3

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = False
        self.write_image_path = './detection/data/detection_char_test/'
        self.show_label       = cfg.TEST.SHOW_LABEL
        self.csv_path         = './detection/mAP/char_pred_test_1600_loss_125.csv'
        self.test_path        = './data/kuzushiji-recognition/test_images/'

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess  = tf.Session(config=self.sess_config)

        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        print('img_size:', image_data.shape)

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes
    
    def get_jp_result(self):
        image_names = os.listdir(self.test_path)
        with open(self.csv_path, 'a') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['image_id','labels'])

            for index, image_name in enumerate(image_names):
                print('=> [%d/%d] predicting image:%s' % (index, len(image_names), image_name))
                image_path = self.test_path + image_name
                image = cv2.imread(image_path)

                bboxes_pr = self.predict(image)
                print('bboxes:',len(bboxes_pr))

                # if self.write_image:
                #     image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                #     cv2.imwrite(self.write_image_path+image_name, image, [int(cv2.IMWRITE_JPEG_QUALITY),30])
            
                image_id = image_name.split('.')[0]
                bb_string = ''
                for bbox in bboxes_pr:
                    
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = self.classes[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(coor)
                    
                    #保存 label x_center y_center
                    # x = int(xmin + (xmax-xmin)/2)
                    # y = int(ymin + (ymax-ymin)/2)
                    # bbox_mess = class_name+' '+str(x)+' '+str(y)+' '

                    #保存 label x y w h形式
                    # w = xmax - xmin
                    # h = ymax - ymin

                    # xmin = int(xmin - 10)
                    # ymin = int(ymin - 5)
                    # xmax = int(xmax + 10)
                    # ymax = int(ymax + 5)
                    
                    # if xmin <= 0:
                    #     xmin = 1
                    # if ymin <= 0:
                    #     ymin = 1
                    # if xmax > image.shape[1]:
                    #    xmax = image.shape[1]
                    # if ymax > image.shape[0]:
                    #    ymax = image.shape[0]

                    w = xmax - xmin
                    h = ymax - ymin
                    bbox_mess = class_name+' '+str(xmin)+' '+str(ymin)+' '+str(w)+' '+str(h)+' '
                    
                    if self.write_image:
                       cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
                    
                    bb_string += bbox_mess
                    #print(bbox_mess)

                f_csv.writerow([image_id, bb_string])
                if self.write_image:
                    cv2.imwrite(self.write_image_path+image_name, image, [int(cv2.IMWRITE_JPEG_QUALITY),30])
                    
if __name__ == '__main__': 
    YoloTest().get_jp_result()




#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File    :   get_dataset_for_yolov3.py
# @Time    :   2019/08/21 16:13:13
# @Author  :   Lai Feng 
# @Contact :   18512447163@163.com
'''
将数据集转化成tensorflow-yolov3需要的格式的数据集
'''
import os
import random
import pandas as pd
import numpy as np
import json
import shutil

TRAIN_CSV_PATH = './data/kuzushiji-recognition/train.csv'

def get_binary_text_dataset():
    '''
    将csv格式训练集转化为yolov3需要的格式，只做文字和非文字检测（'0'为文字）
    保存格式
    img_id xmin,ymin,xmax,ymax,0 xmin,ymin...
    '''
    #csv路径
    df_train = pd.read_csv(TRAIN_CSV_PATH)

    #保存路径
    train_dataset = './data/detect_images/jp_train.txt'
    val_dataset = './data/detect_images/jp_test.txt'
    class_names = './data/detect_images/jp.names'

    #只有文本一类
    with open(class_names, 'w') as wf:
        wf.write('text')

    #划分好的训练集和验证集图片
    train_dir = './data/detect_images/train_images'
    val_dir = './data/detect_images/val_images'
    

    train = os.listdir(train_dir)
    val = os.listdir(val_dir)

    print('train:',len(train),'val:',len(val))

    #写入训练集
    for i in range(len(df_train)):
        print('dealing %d/%d ...' % (i, len(df_train)))
        #img, labels = df_train.values[random.randint(0,len(df_train))]
        img, labels = df_train.values[i]

        img_name = img+'.jpg'

        if img_name in train:
            print('%s in train' % img_name)
            if isinstance(labels, str):
                labels = np.array(labels.split(' ')).reshape(-1, 5) #label x y w h  (x,y)是左上角坐标

                with open(train_dataset, 'a') as wf:
                    wf.write(img_name+' ')
                    for label in labels:
                        class_id = '0'
                        x_min = int(label[1])
                        y_min = int(label[2])
                        x_max = x_min + int(label[3])
                        y_max = y_min + int(label[4])

                        wf.write(str(x_min)+','+str(y_min)+','+str(x_max)+','+str(y_max)+','+str(class_id))
                        wf.write(' ')
                    wf.write('\n')

        #写入测试集
        if img_name in val:
            print('%s in val' % img_name)
            if isinstance(labels, str):
                labels = np.array(labels.split(' ')).reshape(-1, 5) #label x y w h  (x,y)是左上角坐标

                with open(val_dataset, 'a') as wf:
                    wf.write(img_name+' ')
                    for label in labels:
                        class_id = '0'
                        x_min = int(label[1])
                        y_min = int(label[2])
                        x_max = x_min + int(label[3])
                        y_max = y_min + int(label[4])

                        wf.write(str(x_min)+','+str(y_min)+','+str(x_max)+','+str(y_max)+','+str(class_id))
                        wf.write(' ')
                    wf.write('\n')
    

def count_char():
    '''
    统计训练集出现数量大于30的字比例
    '''
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    char_count_dict = {}
    for i in range(len(df_train)):
        print('%d/%d' % (i, len(df_train)))
        img, labels = df_train.values[i]
        #
        if isinstance(labels, str):
            labels = np.array(labels.split(' ')).reshape(-1, 5) #label x y w h  (x,y)是左上角坐标
            for label in labels:
                char = label[0]
                if char in char_count_dict.keys():
                    char_count_dict[char] += 1
                else:
                    char_count_dict[char] = 1

    total = sum(char_count_dict.values())
    left = []
    left_num = 0
    for item in char_count_dict.items():
        if item[1] >= 30:
            left.append(item[0])
            left_num += item[1]

    print('all class num:{} all char num:{} left class num:{} left char num:{} ratio:{}'.format(len(char_count_dict.keys()), total, len(left), left_num, left_num/float(total)))

if __name__ =='__main__':
    get_binary_text_dataset()
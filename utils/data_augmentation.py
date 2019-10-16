# -*- coding: utf-8 -*-

import os

"""
对样本数量少的类别进行重采样类别均衡

MAX: 希望达到的每个类别的最大数量
EXP: 类别增强指数
MIN: 忽略数量非常少的类别
"""

MAX = 2000.0
EXP = 1.0
MIN = 10

def cal_class_mutiple(char_statistics):
    cla_mutiple = {}
    with open(char_statistics, 'r') as fr:
        for line in fr.readlines():
            line = line.strip().split('\t')
            cla, num = line[0], int(line[1])
            if num < MAX and num > 0:
                base_mutiple = int((int(MAX / num)) ** EXP)
            else:
                base_mutiple = 1

            if num > MIN:
                cla_mutiple[cla] = int(base_mutiple)
    return cla_mutiple


def copy_image_path_to_mutiple(cla_mutiple, gt_txt_path, mutiple_gt_txt_path):
    with open(gt_txt_path, 'r') as fr:
        num = 0
        for line in fr.readlines():
            image_name, cla_text = line.strip().split(' ')
            try:
                num += 1
                print(num, image_name)
                for i in range(cla_mutiple[cla_text]):
                    with open(mutiple_gt_txt_path, 'a+') as fw:
                        fw.write(image_name)
                        fw.write(' ')
                        fw.write(cla_text)
                        fw.write('\n')
            except:
                pass



if __name__ == '__main__':
    char_statistics = './data/record/char_statistics.txt'
    croped_image_train_gt_path = './data/rec_images/train_gt.txt'
    mutiple_image_gt_path = './data/rec_images/mutiply_train_gt_2000_10_10.txt'
    cla_mutiple = cal_class_mutiple(char_statistics)
    copy_image_path_to_mutiple(cla_mutiple, croped_image_train_gt_path, mutiple_image_gt_path)

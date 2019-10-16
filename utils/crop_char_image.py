import os
import csv
import re
import random
from PIL import Image

# 根据文字检测框标签裁剪得到字符图像，并划分训练集与验证集

MARGIN_X = 10
MARGIN_Y = 8

gt_file = '/data/kuzushiji-recognition/train.csv'
image_dir = '/data/kuzushiji-recognition/train_images/'
croped_image_dir = './data/rec_images/'
croped_image_train_gt_path = './data/rec_images/train_gt.txt'
croped_image_val_gt_path = './data/rec_images/val_gt.txt'
croped_image_all_gt_path = './data/rec_images/all_gt.txt'
valid_image_dir = './data/detect_images/valid_images/' # 去除检测验证集图像


def croped_char_image(csv_path, image_dir, croped_image_dir, valid_image_dir):
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            image_name = row[0]
            print(image_name)
            if os.path.exists(os.path.join(valid_image_dir, image_name + '.jpg')):
                print('=============pass=============')
                continue
            image_path = os.path.join(image_dir, image_name + '.jpg')
            img = Image.open(image_path)
            if len(row) == 2:
                predictions_list = row[1].split('U+')
                for index, string in enumerate(predictions_list):
                    if string:
                        string = 'U+' + string[:-1]
                        text, cx, cy, w, h = string.split(' ')
                        cx, cy, w, h = int(cx), int(cy), int(w), int(h)
                        bbox_xmin = cx - MARGIN_X
                        bbox_xmax = cx+w + MARGIN_X
                        bbox_ymin = cy - MARGIN_Y
                        bbox_ymax = cy+h + MARGIN_Y
                        bbox = [bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]
                        croped_img = img.crop(bbox)
                        rand = random.randint(0,100)
                        if rand <= 98:
                            save_path = os.path.join(croped_image_dir, 'train/' + image_name + '_{}.jpg'.format(index))
                            with open(croped_image_train_gt_path, 'a+') as fw:
                                fw.write(image_name + '_{}.jpg'.format(index))
                                fw.write(' ')
                                fw.write(text)
                                fw.write('\n')
                        else:
                            save_path = os.path.join(croped_image_dir, 'val/' + image_name + '_{}.jpg'.format(index))
                            with open(croped_image_val_gt_path, 'a+') as fw:
                                fw.write(image_name + '_{}.jpg'.format(index))
                                fw.write(' ')
                                fw.write(text)
                                fw.write('\n')
                        croped_img.save(save_path)



if __name__ == '__main__':
    croped_char_image(gt_file, image_dir, croped_image_dir, valid_image_dir)
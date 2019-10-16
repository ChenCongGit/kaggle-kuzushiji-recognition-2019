# -*- coding: utf-8 -*-

import os
import csv
import re

# 对数据集训练数据类别统计，字符解码等

gt_file = '/data/kuzushiji-recognition/train.csv'
dict_file = '/data/kuzushiji-recognition/unicode_translation.csv'
min_num = 10
output_path = './data/record/char_statistics.txt'

write_keys_path = './data/record/keys_10.txt'
read_keys_path = './data/record/keys_10.txt'

key = b'U+98E9'

def read_gt(csv_path):
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        label_text_list = []
        next(csv_reader)
        for line in csv_reader:
            image_name, gt_line = line
            print(image_name)

            gt_list = gt_line.strip().split(' ')
            label_text = [gt for gt in gt_list if re.match(r"U\+[A-Z0-9]{4,5}", gt)]

            label_text_list.extend(label_text)
    return label_text_list


def read_dict(csv_path):
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        char_dict = {}
        birth_header = next(csv_reader)
        for line in csv_reader:
            char_keys = line[0]
            char_dict[char_keys] = 0
    return char_dict


def char_statistics(label_text_list, char_dict):
    for label_text in label_text_list:
        try:
            char_dict[label_text] += 1
        except KeyError:
            char_dict.update({label_text,0})

    chars = sorted(char_dict.items(), key=lambda x:x[1], reverse=True)
    return chars


def write_txt(chars, output_path):
    with open(output_path, 'w') as fw:
        for key, value in chars:
            fw.write(key + '\t' + str(value) + '\n')


def cal_percent(chars, min_num):
    large = 0.0
    all = 0.0
    print(chars)
    for key, value in chars:
        if value >= min_num:
            large += value
        all += value
    return large/all


def get_keys(keys_path, chars):
    with open(keys_path, 'w') as keys_write:
        for key, value in chars:
            if value >= 30:
                keys_write.write(key + ' ')


def get_keys_from_file(read_keys_path):
  with open(read_keys_path, 'r') as keys_reader:
    keys = keys_reader.read()
    keys = keys.strip().split(' ')
    keys = [key.encode('utf-8') for key in keys]
    return keys


def decode(key):
    key = key.decode('utf-8')
    print(key)


def statistics_char_size(gt_file, char_dict):
    char_height_dict = {}
    char_width_dict = {}
    for char in char_dict.keys():
        char_height_dict[char] = {}
        char_width_dict[char] = {}
    with open(gt_file) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            image_name = row[0]
            if len(row) == 2:
                predictions_list = row[1].split('U+')
                for string in predictions_list:
                    if string:
                        string = 'U+' + string[:-1]
                        text, cx, cy, h, w = string.split(' ')
                        if text in char_dict.keys():
                            if h in char_height_dict[text].keys():
                                char_height_dict[text][h] += 1
                            else:
                                char_height_dict[text][h] = 1
                            if w in char_width_dict[text].keys():
                                char_width_dict[text][w] += 1
                            else:
                                char_width_dict[text][w] = 1
                        else:
                            char_height_dict.update({text, {}})
                            char_width_dict.update({text, {}})
    return char_height_dict, char_width_dict


if __name__ == '__main__':
    # class statistics
    label_text_list = read_gt(gt_file)
    char_dict = read_dict(dict_file)
    chars = char_statistics(label_text_list, char_dict)
    write_txt(chars, output_path)
    # percent = cal_percent(chars, min_num)
    
    # write class key
    # get_keys(write_keys_path, chars)

    # statistics different chars height and width
    # char_height_dict, char_width_dict = statistics_char_size(gt_file, char_dict)

    # read class-num from keys file 
    # keys = get_keys_from_file(read_keys_path)

    # unicode char decoder
    # decode(key)
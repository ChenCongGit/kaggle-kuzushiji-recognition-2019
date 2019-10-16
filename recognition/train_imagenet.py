# -*- coding: utf-8 -*-

import tensorflow as tf
import time
import os
import sys
sys.path.append('./')
import re
import numpy as np

import recognition.resnet as resnet
from recognition.image_processing import image_preprocessing
from recognition.resnet_train import train

flags = tf.app.flags
tf.app.flags.DEFINE_string('data_dir', './data/rec_images/', 'imagenet dir')
tf.app.flags.DEFINE_string('keys_path', './data/record/char_statistics.txt', '')
FLAGS = flags.FLAGS


def get_keys(keys_path):
    with open(keys_path, 'r') as fr:
        text_keys = {}
        label_keys = {}
        for index, line in enumerate(fr.readlines()):
            text, num_sample = line.strip().split('\t')
            if int(num_sample) >= 10:
                text_keys[text] = index
                label_keys[index] = text
        return text_keys, label_keys


def file_list(data_dir):
    dir_txt = data_dir + "mutiply_train_gt_2000_10_10.txt"
    filenames = []
    labels = []
    with open(dir_txt, 'r') as f:
        for line in f.readlines():
            filename, label = line.strip().split(' ')
            image_path = os.path.join(data_dir, 'train/{}'.format(filename))
            filenames.append(image_path)
            labels.append(label)
    return filenames, labels


def text_to_label(texts, files):
    labels = []
    new_texts = []
    new_files = []
    text_keys, label_keys = get_keys(FLAGS.keys_path)
    for index, text in enumerate(texts):
        try:
            labels.append(text_keys[text])
            new_texts.append(text)
            new_files.append(files[index])
        except:
            pass
    return new_texts, labels, new_files


def load_data(data_dir):
    print("listing files in", data_dir)
    start_time = time.time()
    files, texts = file_list(data_dir)
    texts, labels, files = text_to_label(texts, files)
    duration = time.time() - start_time
    print("took %f sec" % duration)
    return files, labels


def distorted_inputs():
    files, labels = load_data(FLAGS.data_dir)
    print(len(files), len(labels))
    filename, label_index = tf.train.slice_input_producer([files, labels], shuffle=True)

    num_preprocess_threads = 4
    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        image_buffer = tf.read_file(filename)

        bbox = []
        train = True
        image = image_preprocessing(image_buffer, bbox, train, thread_id)
        images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.shuffle_batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2 * num_preprocess_threads * FLAGS.batch_size + 10000,
        min_after_dequeue=10000,
        seed=1234)

    height = 128
    width = 128
    depth = 3

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

    return images, tf.reshape(label_index_batch, [FLAGS.batch_size])


def main(_):
    images, labels = distorted_inputs()
    print(images, labels)

    logits = resnet.inference(images,
                       num_classes=2029,
                       is_training=True,
                       bottleneck=True,
                       num_blocks=[3, 4, 6, 3])
    print(logits)
    train(logits, images, labels)


if __name__ == '__main__':
    tf.app.run()

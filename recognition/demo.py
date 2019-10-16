# -*- coding: utf-8 -*-

# 测试验证集图像，单张图片推理

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('./')
import cv2
import csv
import pickle
import random
from PIL import Image

import recognition.resnet as resnet
from recognition.train_imagenet import get_keys
import recognition.image_processing as image_processing


def init_sess(checkpoint_path):

    image_buffer = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name="input_image")
    
    reprocessed_image_tensor = image_processing.image_preprocessing(image_buffer, 1, False)

    # expand 0-dim
    image_tensor = tf.expand_dims(reprocessed_image_tensor, 0)

    # model inference
    logits = resnet.inference(image_tensor,
                       num_classes=2029,
                       is_training=False,
                       bottleneck=True,
                       num_blocks=[3, 4, 6, 3])

    # prediction
    predictions = tf.nn.softmax(logits)

    # fetches and feed
    fetches = {
        'prediction': predictions
    }
    feed = {'input_image_tensor': image_buffer}

    # saver
    saver = tf.train.Saver(tf.global_variables())
    checkpoint = checkpoint_path

    # sess
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.15
    sess = tf.Session(config=sess_config)
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()])
    saver.restore(sess, checkpoint)
    return sess, fetches, feed


def inference(image, sess, fetches, feed):
    """
    resnet网络推理, 
    image: opencv array,64*64*3大小整型
    """
    sess_outputs = sess.run(fetches, feed_dict={feed['input_image_tensor']:image})
    prediction = sess_outputs['prediction']
    return prediction


def postprocess(prediction, keys_path):
    """
    prediction是inference经过softmax输出的[B, C] one-hot 编码数组，将他转化为具体类别
    """
    predict = np.argmax(prediction, axis=1)
    _, label_keys = get_keys(keys_path)
    predict_text = label_keys[predict[0]]
    return predict_text


if __name__ == '__main__':
    val_images_dir = './data/rec_images/val/'
    val_gt_txt_path = './data/rec_images/val_gt.txt'
    checkpoint_path = './recognition/logdir/log/model.ckpt-533501'
    keys_path = './data/record/char_statistics.txt'
    prediction_dict = {}
    sess, fetches, feed = init_sess(checkpoint_path)
    save_num = 0

    with open(keys_path, 'r') as fr:
        label_list = []
        for index, line in enumerate(fr.readlines()):
            text, num_sample = line.strip().split('\t')
            if int(num_sample) >= 10:
                label_list.append(text)

    dir_list = [dir for dir in os.listdir(val_images_dir)]
    random.shuffle(dir_list)
    for dir in list(dir_list):
        save_num += 1
        print(save_num, dir)
        image_name = os.path.join(val_images_dir, dir)
        image = Image.open(image_name)
        image_data = np.asarray(image)/255.0
        prediction = inference(image_data, sess, fetches, feed)
        predict_result = postprocess(prediction, keys_path)
        prediction_dict[dir] = predict_result

        # if save_num >= 10000:
        #     break

    print('预测完成----------------------------------')

    with open(val_gt_txt_path, 'r') as fr:
        count_images = 0.0
        count_correct = 0.0
        count_images_with_ignore = 0.0
        for line in fr.readlines():
            image_name, label = line.strip().split(' ')
            if image_name in prediction_dict.keys():
                count_images += 1
                # 忽略的小于50的标签类别
                if label in label_list:
                    count_images_with_ignore += 1
                    print(image_name)
                    if prediction_dict[image_name] == label:
                        count_correct += 1
        print(count_correct, count_images, count_images_with_ignore)
        print(count_correct/count_images, count_correct/count_images_with_ignore)


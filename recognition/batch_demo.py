# -*- coding: utf-8 -*-

# 测试集图像推理

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('./')
import cv2
import time
import csv
import pickle
import random
from PIL import Image

import recognition.resnet as resnet
from recognition.train_imagenet import get_keys
import recognition.image_processing as image_processing


def batch_init_sess(checkpoint_path):
    batch_images = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name="input_image")

    # model inference
    batch_logits = resnet.inference(batch_images,
                       num_classes=2029,
                       is_training=False,
                       bottleneck=True,
                       num_blocks=[3, 4, 6, 3])

    # prediction
    batch_predictions = tf.nn.softmax(batch_logits)

    # fetches and feed
    fetches = {
        'prediction': batch_predictions
    }
    feed = {'input_image_tensor': batch_images}

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


def batch_inference(batch_images, sess, fetches, feed):
    """
    resnet网络推理, 
    image: opencv array, B*64*64*3大小整型
    """
    sess_outputs = sess.run(fetches, feed_dict={feed['input_image_tensor']:batch_images})
    prediction = sess_outputs['prediction']
    return prediction


def tta_inference(batch_images, sess, fetches, feed):
    # Test Time Augmentation (TTA)
    predictions = []
    for i in range(3):
        print('Start TTA %d : ' % i)
        # test images augmentation
        batch_images = test_images_aug(batch_images, i)

        sess_outputs = sess.run(fetches, feed_dict={feed['input_image_tensor']:batch_images})
        prediction = sess_outputs['prediction']

        predictions.append(prediction)
    tta_prediction = np.stack(predictions, axis=0) # [N, B, C]

    # TTA SUM
    # prediction = np.mean(tta_prediction, axis=0, keepdims=False)
    # TTA MAX
    max_prob = np.max(tta_prediction, axis=2, keepdims=False) # [N, B]
    max_indices = np.argmax(max_prob, axis=0) # [B]
    prediction = []
    for j in range(max_indices.shape[0]):
        one_prediction = tta_prediction[max_indices[j], j, :]
        prediction.append(one_prediction)
    prediction = np.stack(prediction, axis=0)
    return prediction


def test_images_aug(batch_image, k):
    """
    测试时多尺度，图像增强预测
    """
    batch = []
    for i in range(batch_image.shape[0]):
        # multi scale
        scale_list = [100, 128, 156]
        image = cv2.resize(batch_image[i], (scale_list[k], scale_list[k]))
        batch.append(image)
    batch_image = np.stack(batch, axis=0)
    
    return batch_image


def batch_postprocess(prediction, label_keys):
    """
    prediction是inference经过softmax输出的[B, C] one-hot 编码数组，将他转化为具体类别
    """
    predict_text_list = []
    for i in range(prediction.shape[0]):
        if np.max(prediction[i]) > 0.85:
            predict = np.argmax(prediction[i])
            predict_text = label_keys[predict]
        else:
            predict_text = "-1"
        predict_text_list.append(predict_text)
    return predict_text_list


def batch_postprocess_with_nms(prediction, bboxes,  label_keys):
    """
    结合检测框和识别结果进行nms后处理
    """
    scores = np.max(prediction, axis=1, keepdims=True)
    classes = np.argmax(prediction, axis=1)[:,np.newaxis]
    bboxes_with_class = np.concatenate([bboxes, scores, classes], axis=1)
    best_predictions = nms(bboxes_with_class, iou_threshold=0.4)

    predict_text_list = []
    bbox_list = []
    for i in range(len(best_predictions)):
        if np.max(best_predictions[i][4]) > 0.6:
            predict_text = label_keys[best_predictions[i][-1]]
        else:
            predict_text = "-1"
        predict_text_list.append(predict_text)

        x_min, y_min, x_max, y_max = best_predictions[i][0:4]
        x = int((x_min + x_max) / 2.0)
        y = int((y_min + y_max) / 2.0)
        bbox_list.append([x, y])
    return predict_text_list, bbox_list


def preprocessing(image):
    """
    image: 被裁剪下来的单个字图像,像素值归一化为0-1
    """
    preocess_image = np.copy(image)
    preocess_image = image_padding(preocess_image)
    preocess_image = cv2.resize(preocess_image, (128, 128))
    preocess_image = (preocess_image - 0.5) * 2 # rescale to [-1,1] instead of [0, 1)
    return image, preocess_image


def image_padding(image):
    """
    将图像padding到固定长宽比
    """
    aspect_ratio = 1.0
    h, w = image.shape[0:2]
    if h > aspect_ratio * w:
        pad_left = int((h - w) / 2)
        pad_right = int(h - pad_left - w)
        image = np.pad(image, ((0,0), (pad_left, pad_right), (0,0)), 'constant', constant_values=(0,0))
    elif w > aspect_ratio * h:
        pad_up = int((w - h) / 2)
        pad_down = int(w - pad_up - h)
        image = np.pad(image, ((pad_up,pad_down), (0,0), (0,0)), 'constant', constant_values=(0,0))
    else:
        pass
    return image


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes = []
    cls_bboxes = bboxes

    while len(cls_bboxes) > 0:
        max_ind = np.argmax(cls_bboxes[:, 4])
        best_bbox = cls_bboxes[max_ind]
        best_bboxes.append(best_bbox)
        cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
        weight = np.ones((len(iou),), dtype=np.float32)

        assert method in ['nms', 'soft-nms']

        if method == 'nms':
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0

        if method == 'soft-nms':
            weight = np.exp(-(1.0 * iou ** 2 / sigma))

        cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
        score_mask = cls_bboxes[:, 4] > 0.
        cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def demo(test_cvs_path,
         submission_csv_path,
         checkpoint_path,
         test_images_dir,
         keys_path):
    sess, fetches, feed = batch_init_sess(checkpoint_path)
    
    print("============================图像识别开始==============================")

    # get image submission result
    with open(test_cvs_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        image_result_dict = {}
        image_num = 0
        _, label_keys = get_keys(keys_path)
        for row in csv_reader:
            image_num += 1
            image_name = row[0]
            img = Image.open(os.path.join(test_images_dir, image_name + '.jpg'))
            img_array = np.asarray(img)/255.0
            print(image_num, image_name)
            image_result_dict[image_name] = []

            if row[1] != '':
                predictions_list = row[1].split('text')
                image_list = []
                bbox_list = []
                bbox_array_list = []
                for string in predictions_list:
                    if string:
                        string = 'text' + string[:-1]
                        _, cx, cy, w, h = string.split(' ')

                        x_min, x_max = int(cx), int(cx) + int(w)
                        y_min, y_max = int(cy), int(cy) + int(h)
                        bbox_array = np.array([x_min, y_min, x_max, y_max])
                        bbox_array_list.append(bbox_array)

                        # get bbox place
                        x = int((x_min + x_max) / 2.0)
                        y = int((y_min + y_max) / 2.0)
                        bbox_list.append([x, y])

                        # crop and preprocess one image
                        croped_image_array = img_array[y_min:y_max, x_min:x_max, :]
                        _, preocessed_image = preprocessing(croped_image_array)

                        image_list.append(preocessed_image)

                # batch all images in one page
                batch_images = np.stack(image_list, axis=0)
                # batch_bboxes = np.stack(bbox_array_list, axis=0)

                # batch inference and predict
                prediction = batch_inference(batch_images, sess, fetches, feed)
                # prediction = tta_inference(batch_images, sess, fetches, feed)
                predict_result_list = batch_postprocess(prediction, label_keys)
                # predict_result_list, bbox_list = batch_postprocess_with_nms(prediction, batch_bboxes, label_keys)

                assert len(predict_result_list) == len(bbox_list)
                # 保存中心点
                image_result_dict[image_name] = \
                    [[predict_result_list[i], bbox_list[i][0], bbox_list[i][1]] \
                        for i in range(len(predict_result_list)) if predict_result_list[i] != "-1"]

        csv_file.close()

    print("============================图像识别完成==============================")

    # write submission result to csv
    with open(submission_csv_path, "w") as sub_csv_file:
        writer = csv.writer(sub_csv_file)
        writer.writerows([["image_id", "labels"]])

        for image_name, image_result in image_result_dict.items():
            res = [image_name]
            string = ""
            for line in image_result:
                string += line[0] + " " + str(line[1]) + " " + str(line[2]) + " "
            res.append(string)
            writer.writerows([res])

        sub_csv_file.close()
    print("============================结果写入成功==============================")


if __name__ == '__main__':
    """
    读取csv文件检测框并识别
    """
    test_cvs_path = './result/detect_result/char_pred_test_loss_125.csv'
    submission_csv_path = './result/submission/char_pred_test_loss_125_533501_2029_sub_06_128.csv'
    test_images_dir = '/data/kuzushiji-recognition/test_images/'
    checkpoint_path = './recognition/logdir/log/model.ckpt-533501'
    keys_path = './data/record/char_statistics.txt'

    # read test cvs file and crop images
    demo(test_cvs_path, 
         submission_csv_path,
         checkpoint_path, 
         test_images_dir,  
         keys_path)
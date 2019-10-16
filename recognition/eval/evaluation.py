"""
Python equivalent of the Kuzushiji competition metric (https://www.kaggle.com/c/kuzushiji-recognition/)
Kaggle's backend uses a C# implementation of the same metric. This version is
provided for convenience only; in the event of any discrepancies the C# implementation
is the master version.

Tested on Python 3.6 with numpy 1.16.4 and pandas 0.24.2.

Usage: python f1.py --sub_path [submission.csv] --solution_path [groundtruth.csv]
"""

import argparse
import multiprocessing
import sys

import numpy as np
import pandas as pd


def define_console_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    return parser


def score_page(preds, truth):
    """
    Scores a single page.
    Args:
        preds: prediction string of labels and center points.
        truth: ground truth string of labels and bounding boxes.
    Returns:
        True/false positive and false negative counts for the page
    """
    #print(len(preds), len(truth))
    tp = 0
    fp = 0
    fn = 0
    bbox_tp = 0
    bbox_fp = 0
    bbox_fn = 0

    truth_indices = {
        'label': 0,
        'X': 1,
        'Y': 2,
        'Width': 3,
        'Height': 4
    }
    preds_indices = {
        'label': 0,
        'X': 1,
        'Y': 2
    }

    if pd.isna(truth) and pd.isna(preds):
        return {'tp': tp, 'fp': fp, 'fn': fn, 'bbox_tp': bbox_tp, 'bbox_fp': bbox_fp, 'bbox_fn': bbox_fn}

    if pd.isna(truth):
        fp += len(preds.split(' ')) // len(preds_indices)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'bbox_tp': bbox_tp, 'bbox_fp': bbox_fp, 'bbox_fn': bbox_fn}

    if pd.isna(preds):
        fn += len(truth.split(' ')) // len(truth_indices)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'bbox_tp': bbox_tp, 'bbox_fp': bbox_fp, 'bbox_fn': bbox_fn}

    truth = truth.strip().split(' ')
    #print('truth len:', len(truth))
    if len(truth) % len(truth_indices) != 0:
        raise ValueError('Malformed solution string')
    truth_label = np.array(truth[truth_indices['label']::len(truth_indices)])
    truth_xmin = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
    truth_ymin = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
    truth_xmax = truth_xmin + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
    truth_ymax = truth_ymin + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

    preds = preds.strip().split(' ')
    #print('pred len:', len(preds))
    # print(len(preds))
    if len(preds) % len(preds_indices) != 0:
        raise ValueError('Malformed prediction string')
    preds_label = np.array(preds[preds_indices['label']::len(preds_indices)])
    preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
    preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)
    preds_unused = np.ones(len(preds_label)).astype(bool)
    preds_unused_bbox = np.ones(len(preds_label)).astype(bool)

    for xmin, xmax, ymin, ymax, label in zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label):
        # Matching = point inside box & character same & prediction not already used
        matching_bbox = (xmin < preds_x) & (xmax > preds_x) & (ymin < preds_y) & (ymax > preds_y) & preds_unused
        if matching_bbox.sum() == 0:
            bbox_fn += 1
        else:
            bbox_tp += 1
            preds_unused_bbox[np.argmax(matching_bbox)] = False

    for xmin, xmax, ymin, ymax, label in zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label):
        # Matching = point inside box & character same & prediction not already used
        matching_bbox = (xmin < preds_x) & (xmax > preds_x) & (ymin < preds_y) & (ymax > preds_y) & preds_unused
        matching = matching_bbox & (preds_label == label)
        if matching.sum() == 0:
            fn += 1
        else:
            tp += 1
            preds_unused[np.argmax(matching_bbox)] = False

    bbox_fp += preds_unused_bbox.sum()
    fp += preds_unused.sum()
    return {'tp': tp, 'fp': fp, 'fn': fn, 'bbox_tp': bbox_tp, 'bbox_fp': bbox_fp, 'bbox_fn': bbox_fn}


def kuzushiji_f1(sub, solution):
    """
    Calculates the competition metric.
    Args:
        sub: submissions, as a Pandas dataframe
        solution: solution, as a Pandas dataframe
    Returns:
        f1 score
    """
    # print(sub['image_id'].values, solution['image_id'].values)
    if not all(sub['image_id'].values == solution['image_id'].values):
        raise ValueError("Submission image id codes don't match solution")

    pool = multiprocessing.Pool()
    results = pool.starmap(score_page, zip(sub['labels'].values, solution['labels'].values))
    pool.close()
    pool.join()

    tp = sum([x['tp'] for x in results])
    fp = sum([x['fp'] for x in results])
    fn = sum([x['fn'] for x in results])

    bbox_tp = sum([x['bbox_tp'] for x in results])
    bbox_fp = sum([x['bbox_fp'] for x in results])
    bbox_fn = sum([x['bbox_fn'] for x in results])


    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    bbox_precision = bbox_tp / (bbox_tp + bbox_fp)
    bbox_recall = bbox_tp / (bbox_tp + bbox_fn)

    print(precision, recall, bbox_precision, bbox_recall)
    if precision > 0 and recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python {} [-h] --sub_path SUB_PATH --solution_path SOLUTION_PATH'.format(sys.argv[0]))
        exit()
    parser = define_console_parser()
    shell_args = parser.parse_args()
    sub = pd.read_csv(shell_args.sub_path)
    solution = pd.read_csv(shell_args.solution_path)
    # print(sub, solution)
    sub = sub.sort_values(by=['image_id'],na_position='first')
    solution = solution.sort_values(by=['image_id'],na_position='first')
    print(sub, solution)
    sub = sub.rename(columns={'rowId': 'image_id', 'PredictionString': 'labels'})
    solution = solution.rename(columns={'rowId': 'image_id', 'PredictionString': 'labels'})
    score = kuzushiji_f1(sub, solution)
    # import csv
    print('F1 score of: {0}'.format(score))
    # sub = pd.read_csv('/home/AI/chencong/tensorflow-resnet/result/submission/submission_test.csv')
    # sub = sub.sort_values(by=['image_id'],na_position='first')
    # with open('./new.csv', 'a') as wf:
    #     writer = csv.writer(wf)
    #     writer.writerows([["image_id", "labels"]])

    #     for i in range(len(sub)):
    #         image_id, labels = sub.values[i]
    #         if not pd.isna(labels):
    #             labels = labels.strip()
    #         else:
    #             labels = ""

    #         writer.writerows([[image_id, labels]])
        
    
    

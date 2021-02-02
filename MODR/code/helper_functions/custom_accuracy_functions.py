import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import applications
from helper_functions.custom_loss_functions import get_weighted_loss
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import errno
import cv2


def get_class_weights(data, columns, cw_setting):
    if cw_setting is None:
        print('Class Weights are set to None.')
        return None
    elif cw_setting == 'auto':
        print('Class Weights are set to "auto".')
        return 'auto'
    elif cw_setting == 'mc':
        summed_arr = np.sum(data[columns].values, axis=0)
        class_weights = np.max(summed_arr) / summed_arr
        return {idx: w for idx, w in zip(range(len(class_weights)), class_weights)}
    elif cw_setting == 'ml':
        number_dim = np.shape(data[columns].values)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            weights[i] = compute_class_weight('balanced', [0., 1.], data[columns].values[:, i])
        return weights
    else:
        raise ValueError("Specified class weight setting: {} not found.".format(cw_setting))


def count_ones(y_true, y_pred):
    y_pred = K.round(y_pred)
    return K.sum(y_pred) - K.sum(y_true)


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall/Sensitivity metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def specificity(y_true, y_pred):
    """Specificity/True Negative Rate"""
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    # tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    # fn = K.sum(y_pos * y_pred_neg)
    tnr = tn/(tn+fp+K.epsilon())
    return tnr

### NOT FOR TRAINING ###
def top_n_accuracy(y_true, y_pred, n=3, labels=None):
    top_n = np.argsort(y_pred)[:, -1*n:]
    y_pred = np.round(y_pred)
    # Generate top-3 predictions
    top_n_metric = []
    tmp = np.zeros(y_pred.shape)
    for t3, y_p, y_t, i in zip(top_n, y_pred, y_true, range(y_pred.shape[0])):
        top_n_metric.append(y_p[t3] == y_t[t3])
        # Method 1
        # tmp[i, t3] = 1
        # Method 2
        tmp[i, t3] = y_p[t3]
    top_n_metric = np.array(top_n_metric)
    print("Average: {}".format(np.average(top_n_metric)))
    print(classification_report(y_true, tmp, target_names=labels))
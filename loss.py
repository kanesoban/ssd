import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from sklearn.preprocessing import OneHotEncoder

from utils import batch_iou


def get_smoothL1_loss(predictions, labels, object_matches, delta=1.0):
    ''' Calculate smooth L1 loss
    Args:
        predictions (batch size, dimensions)
        labels (batch size, dimensions)
    Returns:
        smooth L1 loss according to https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf
    '''
    #assert(predictions.shape == labels.shape)
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    # Simpler L1 loss
    #return tf.reduce_mean(math_ops.abs(math_ops.subtract(predictions, labels)))
    l1_losses = tf.reshape(object_matches, (-1, object_matches.shape[1])) * math_ops.reduce_sum(math_ops.abs(math_ops.subtract(predictions, labels)), axis=-1)
    condition = tf.less(l1_losses, delta)
    smoothL1_loss = tf.where(condition, 0.5 * (l1_losses**2), l1_losses - 0.5)
    return tf.reduce_mean(smoothL1_loss)
    #return tf.reduce_mean(1.0/tf.math.reduce_sum(object_matches, axis=1) * smoothL1_loss)


def get_relative_gt_box(ground_truths, default_boxes):
    ''' Calculate relative ground truth center coordinates and size
    Args:
        ground_truths: ground truth boxes (number of priors, 4)
        default_boxes: prior boxes (number of priors, 4)
    Returns:
        relative (to the prior boxes) ground truth box center coordinates and sizes (number of priors, 4)
    '''
    rel_ground_truths = np.empty_like(ground_truths)
    rel_ground_truths[:, :2] = (ground_truths[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:]
    rel_ground_truths[:, 2:] = np.log(ground_truths[:, 2:] / default_boxes[:, 2:])
    return rel_ground_truths


def to_iou_coordinates(array):
    iou_coordinates = np.empty_like(array)
    iou_coordinates[:, :, :2] = array[:, :, :2] - array[:, :, 2:]/2
    iou_coordinates[:, :, 2:] = array[:, :, :2] + array[:, :, 2:]/2
    return iou_coordinates


def calculate_array_matches(ground_truths, predictions, iou_threshold=0.5):
    ious = batch_iou(to_iou_coordinates(ground_truths), to_iou_coordinates(predictions))
    return np.where(iou_threshold < ious, 1.0, 0.0).astype(np.float32)


def calculate_matches(ground_truths, predictions, iou_threshold=0.5):
    object_matches = tf.py_func(calculate_array_matches, [ground_truths, predictions], tf.float32)
    object_matches.set_shape((ground_truths.shape[0], ground_truths.shape[1], 1))
    return object_matches


def get_localization_loss(ground_truths, object_matches):
    def localization_loss(_, predictions):
        ''' Calculate ssd localization loss
        Args:
            predictions: (number of total priors, 4)
            ground_truths: ground truth boxes (number of total priors, 4)
        Returns:
            ssd localization loss
        '''
        return get_smoothL1_loss(labels=ground_truths, predictions=predictions, object_matches=object_matches)

    return localization_loss


def confidence_loss(labels, logits):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)


def ssd300_loss(ground_truth_bb, object_matches, alpha=1.0):
    losses = {
        'confidence': confidence_loss,
        'localization': get_localization_loss(ground_truth_bb, object_matches),
    }
    loss_weights = {'confidence': 1.0, 'localization': alpha}
    return losses, loss_weights


def voc_class_to_ohe(matches, class_ids, n_classes):
    # Add "background" category to labels where there was no match with the designated prior box
    indicators = class_ids.copy()
    indicators[~matches] = 0

    # Convert Pascal voc labels to one hot encoding with bakcground class 
    ohe = OneHotEncoder()
    ohe.fit(np.array(range(n_classes + 1)).reshape(-1, 1))
    return ohe.transform(indicators).toarray().reshape()

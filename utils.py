import numpy as np
import tensorflow as tf


def get_single_anchor_box(i, j, scale, aspect_ratio, feature_map_width, img_height, img_width):
    box_width = scale * np.sqrt(aspect_ratio)
    box_height = scale / np.sqrt(aspect_ratio)
    center = ((i + 0.5)/feature_map_width, (j + 0.5)/feature_map_width)
    upper_left = (center[0] - box_width/2, center[1] - box_height/2)
    lower_right = (center[0] + box_width/2, center[1] + box_height/2)
    anchor = np.array([center[0], center[1], box_width, box_height])
    x = np.array([upper_left[0], upper_left[0], lower_right[0], lower_right[0], upper_left[0]]) * img_height
    y = np.array([upper_left[1], lower_right[1], lower_right[1], upper_left[1], upper_left[1]]) * img_width
    return anchor, x, y


'''
def create_anchor_boxes(aspect_ratios, scales, img_size):
    anchors = []
    for scale in scales:
        feature_map_width = int(1.0 / scale)
        for i in range(feature_map_width):
            for j in range(feature_map_width):
                for aspect_ratio in aspect_ratios:
                    anchor, x, y = get_single_anchor_box(i, j, scale, aspect_ratio, feature_map_width, img_size, img_size)
                    anchors.append(anchor)
                #TODO: +1 aspect ratio
    return np.array(anchors)
'''


def create_anchor_boxes(output_shape, aspect_ratios, scale, img_size):
    feature_map_width, _, priors_per_cell, features_per_box = output_shape
    anchors = []
    for i in range(feature_map_width):
        for j in range(feature_map_width):
            for aspect_ratio in aspect_ratios:
                anchor, x, y = get_single_anchor_box(i, j, scale, aspect_ratio, feature_map_width, img_size, img_size)
                anchors.append(anchor)
            #TODO: +1 aspect ratio
    return np.array(anchors)

'''
def create_anchor_boxes(output_shapes, aspect_ratios, scales, img_size):
    assert(len(output_shapes) == len(scales))
    anchors = []
    for scale_index, scale in enumerate(scales):
        anchors += create_anchor_boxes(output_shapes[scale_index], aspect_ratios, scale, img_size)
    return np.array(anchors)
'''


def create_scales(s_min, s_max, num_scales):
    scales = []
    for k in range(1, num_scales+1):
        scales.append(s_min + (s_max - s_min) / (num_scales - 1) * (k-1))
    return scales


def batch_iou(a, b, epsilon=1e-8):
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def iou_tf(a, b):
    return tf.py_func(batch_iou, [a, b], tf.float32)


def iou(associations, name1='prediction', name2='gt', epsilon=1e-8):
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = associations[['x0_' + name1, 'x0_' + name2]].max(axis=1)
    y1 = associations[['y0_' + name1, 'y0_' + name2]].max(axis=1)
    x2 = associations[['x1_' + name1, 'y1_' + name2]].min(axis=1)
    y2 = associations[['y1_' + name1, 'y1_' + name2]].min(axis=1)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (associations['x1_' + name1] - associations['x0_' + name1]) * (associations['y1_' + name1] - associations['y0_' + name1])
    area_b = (associations['x1_' + name2] - associations['x0_' + name2]) * (associations['y1_' + name2] - associations['y0_' + name2])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou

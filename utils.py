import pandas as pd
import numpy as np


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

import pandas as pd
import numpy as np


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


def iou(associations, epsilon=1e-8):
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = associations[['x0_prior', 'x0_gt']].max(axis=1)
    y1 = associations[['y0_prior', 'y0_gt']].max(axis=1)
    x2 = associations[['x1_prior', 'y1_gt']].min(axis=1)
    y2 = associations[['y1_prior', 'y1_gt']].min(axis=1)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (associations['x1_prior'] - associations['x0_prior']) * (associations['y1_prior'] - associations['y0_prior'])
    area_b = (associations['x1_gt'] - associations['x0_gt']) * (associations['y1_gt'] - associations['y0_gt'])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou

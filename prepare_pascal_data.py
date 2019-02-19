import h5py
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.preprocessing import OneHotEncoder
from gluoncv.data import VOCDetection

from ssd import SSD300
from utils import iou


MAX_SCALE = 0.9
MIN_SCALE = 0.2
NUM_SCALES = 6
IMG_SIZE = 300
CHANNELS = 3
NUM_VOC_CLASSES = 20


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=None, help='Path to Pascal VOC dataset root')
    parser.add_argument('--out_path', default='data.h5', help='Path to output')
    parser.add_argument('--train', default=True, type=bool, help='Train or test dataset')
    parser.add_argument('--single_data', default=None, type=int, help='Select single datapoint')
    return parser.parse_args()


def create_class_encoder():
    ohe = OneHotEncoder()
    ohe.fit(np.array(range(NUM_VOC_CLASSES + 1)).reshape(-1, 1))
    return ohe


def get_anchor_boxes():
    model = SSD300()
    return model.anchors


def add_corner_coordinates(df):
    df['x0'] = df['center_x'] - df['w'] / 2
    df['y0'] = df['center_y'] - df['h'] / 2
    df['x1'] = df['center_x'] + df['w'] / 2
    df['y1'] = df['center_y'] + df['h'] / 2


def bounding_box_to_df(bounding_boxes, table_type):
    bounding_boxes = pd.DataFrame(bounding_boxes, columns=['center_x', 'center_y', 'w', 'h'])
    bounding_boxes[table_type + '_id'] = range(len(bounding_boxes))
    bounding_boxes['temp'] = True
    add_corner_coordinates(bounding_boxes)
    return bounding_boxes


def match_ground_truths_to_priors(bounding_boxes, class_ids, anchors, iou_threshold=0.5):
    pred_type = 'prediction'
    gt_type = 'gt'

    anchors = bounding_box_to_df(anchors, pred_type)

    ground_truth_boxes = bounding_box_to_df(bounding_boxes, gt_type)
    ground_truth_boxes['class_id'] = class_ids

    associations = pd.merge(anchors, ground_truth_boxes, how='outer', suffixes=('_' + pred_type, '_' + gt_type), on='temp')
    associations = associations.drop(columns=['temp'])
    associations['iou'] = iou(associations)
    associations['match'] = (associations['iou'] > iou_threshold)

    best_priors = associations.iloc[associations.groupby('prediction_id').apply(lambda g: g.iou.idxmax())]
    return best_priors[['center_x_gt', 'center_y_gt', 'w_gt', 'h_gt']], best_priors['iou'], best_priors['class_id']


def preprocess_image(image):
    return cv2.resize(image.asnumpy(), (IMG_SIZE, IMG_SIZE)) / 255.0


def convert_to_anchor_coordinates(image, bounding_boxes):
    image_width = image.shape[1]
    image_height = image.shape[0]
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    abs_width = x2 - x1
    abs_height = y2 - y1
    abs_center_x = x1 + abs_width / 2
    abs_center_y = y1 + abs_height / 2
    cx = abs_center_x / image_width
    cy = abs_center_y / image_height
    w = abs_width / image_width
    h = abs_height / image_height
    return np.stack([cx, cy, w, h], axis=1)


def convert(args):
    if args.train:
        splits = [(2007, 'trainval'), (2012, 'trainval')]
    else:
        splits = [(2007, 'test')]

    gluon_dataset = VOCDetection(root=args.data_root, splits=splits)
    if args.single_data:
        rng = [args.single_data]
        num_data = 1
    else:
        rng = range(len(gluon_dataset))
        num_data = len(gluon_dataset)

    with h5py.File(args.out_path, 'w') as f:
        group = f.create_group('main')

        anchors = np.concatenate(get_anchor_boxes(), axis=0)

        num_total_priors = anchors.shape[0]

        anchors_dataset = group.create_dataset('anchors', anchors.shape, dtype='f')
        anchors_dataset[:] = anchors
        image_dataset = group.create_dataset('image', (num_data, IMG_SIZE, IMG_SIZE, 3), dtype='f')
        # 5 = 4 for box coordinates + iou
        bounding_boxes_dataset = group.create_dataset('bounding_box', (num_data, num_total_priors, 4), dtype='f')
        iou_dataset = group.create_dataset('iou', (num_data, num_total_priors, 1))
        class_dataset = group.create_dataset('class', (num_data, num_total_priors, 1), dtype='i')
        #class_dataset = group.create_dataset('class', (num_data, num_total_priors, NUM_VOC_CLASSES + 1), dtype='i')
        #ohe_encoder = create_class_encoder()

        for i, data_index in tqdm(enumerate(rng)):
            image, label = gluon_dataset[data_index]
            bounding_boxes = label[:, :4]
            class_ids = label[:, 4:5]
            image_dataset[i] = preprocess_image(image)
            bounding_boxes = convert_to_anchor_coordinates(image, bounding_boxes)
            bounding_boxes, ious, classes = match_ground_truths_to_priors(bounding_boxes, class_ids, anchors)
            #class_ids = ohe_encoder.transform(class_ids)
            bounding_boxes_dataset[i] = bounding_boxes.values
            iou_dataset[i] = ious.values.reshape((-1, 1))
            class_dataset[i] = classes.values.reshape((-1, 1))


if __name__ == "__main__":
    args = get_args()
    convert(args)

import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras.layers import Layer, Reshape, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

from utils import create_anchor_boxes, create_scales
from loss import ssd300_loss


def mult(shape):
    features = 1
    for d in shape:
        features = features * int(d)
    return int(features)


class Localization(Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors
        self.n_priors = None
        super(Localization, self).__init__(name='localization', **kwargs)

    def build(self, input_shape=None):
        super(Localization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outs = []
        for inp, anchor in zip(inputs, self.anchors):
            outs.append(inp[:, :, -4:] + anchor)
        output = tf.concat(outs, axis=1)
        self.n_priors = int(output.shape[1])
        return output

    def compute_output_shape(self, input_shape):
        return None, self.n_priors, 4


class Confidence(Layer):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        self.n_priors = None
        super(Confidence, self).__init__(name='confidence', **kwargs)

    def build(self, input_shape=None):
        super(Confidence, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outs = []
        for inp in inputs:
            outs.append(inp[:, :, :self.n_classes])
        output = tf.concat(outs, axis=1)
        self.n_priors = int(output.shape[1])
        return output

    def compute_output_shape(self, input_shape):
        return None, self.n_priors, self.n_classes


class SSD300:
    @staticmethod
    def mult(shape):
        features = 1
        for d in shape:
            features = features * int(d)
        return features

    def compute_anchor_shape(self, num_priors, output_tensor):
        features = SSD300.mult(output_tensor.shape[1:])
        feature_map_width = int(np.sqrt(features / (num_priors * (self.classes + 4))))
        new_shape = (feature_map_width, feature_map_width, num_priors, 4 + self.classes)
        assert(SSD300.mult(new_shape) == features)
        return new_shape

    @property
    def output_shapes(self):
        shapes = []
        for output_tensor in self.outputs:
            shapes.append(output_tensor.shape)
        return shapes

    def add_ssd300_layers(self, vgg_output_tensor):
        with tf.variable_scope('ssd_300'):
            anchor_shapes = []
            scale1, scale2, scale3, scale4, scale5, scale6 = create_scales(self.min_scale, self.max_scale, 6)

            # Classifier + Localizer 1
            #priors_per_cell = 4
            priors_per_cell = 5
            output1_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output1')(vgg_output_tensor)

            anchor_shapes.append(self.compute_anchor_shape(priors_per_cell, output1_tensor))
            self.anchors.append(create_anchor_boxes(anchor_shapes[-1], self.aspect_ratios, scale1, self.img_size))

            num_anchors_in_feature_map = SSD300.mult(anchor_shapes[-1][:-1])
            self.outputs.append(Reshape((num_anchors_in_feature_map, self.classes + 4))(output1_tensor))


            # Convolutional block 1
            block_tensor = layers.Conv2D(1024, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block1_1')(vgg_output_tensor)

            block_tensor = layers.Conv2D(1024, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block1_2')(block_tensor)

            # Classifier + Localizer 2
            #priors_per_cell = 6
            priors_per_cell = 5
            output2_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output2')(block_tensor)

            anchor_shapes.append(self.compute_anchor_shape(priors_per_cell, output1_tensor))
            self.anchors.append(create_anchor_boxes(anchor_shapes[-1], self.aspect_ratios, scale2, self.img_size))

            num_anchors_in_feature_map = SSD300.mult(anchor_shapes[-1][:-1])
            self.outputs.append(Reshape((num_anchors_in_feature_map, self.classes + 4))(output2_tensor))


            # Convolutional block 2
            block_tensor = layers.Conv2D(256, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block2_1')(block_tensor)

            block_tensor = layers.Conv2D(512, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block2_2')(block_tensor)

            # Classifier + Localizer 3
            #priors_per_cell = 6
            priors_per_cell = 5
            output3_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output3')(block_tensor)

            anchor_shapes.append(self.compute_anchor_shape(priors_per_cell, output1_tensor))
            self.anchors.append(create_anchor_boxes(anchor_shapes[-1], self.aspect_ratios, scale3, self.img_size))

            num_anchors_in_feature_map = SSD300.mult(anchor_shapes[-1][:-1])
            self.outputs.append(Reshape((num_anchors_in_feature_map, self.classes + 4))(output3_tensor))

            # Convolutional block 3
            block_tensor = layers.Conv2D(128, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block3_1')(block_tensor)

            block_tensor = layers.Conv2D(256, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block3_2')(block_tensor)

            # Classifier + Localizer 4
            #priors_per_cell = 6
            priors_per_cell = 5
            output4_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output4')(block_tensor)

            anchor_shapes.append(self.compute_anchor_shape(priors_per_cell, output1_tensor))
            self.anchors.append(create_anchor_boxes(anchor_shapes[-1], self.aspect_ratios, scale4, self.img_size))

            num_anchors_in_feature_map = SSD300.mult(anchor_shapes[-1][:-1])
            self.outputs.append(Reshape((num_anchors_in_feature_map, self.classes + 4))(output4_tensor))

            # Convolutional block 4
            block_tensor = layers.Conv2D(128, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block4_1')(block_tensor)

            block_tensor = layers.Conv2D(256, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block4_2')(block_tensor)

            # Classifier + Localizer 5
            #priors_per_cell = 4
            priors_per_cell = 5
            output5_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output5')(block_tensor)

            anchor_shapes.append(self.compute_anchor_shape(priors_per_cell, output1_tensor))
            self.anchors.append(create_anchor_boxes(anchor_shapes[-1], self.aspect_ratios, scale5, self.img_size))

            num_anchors_in_feature_map = SSD300.mult(anchor_shapes[-1][:-1])
            self.outputs.append(Reshape((num_anchors_in_feature_map, self.classes + 4))(output5_tensor))

            # Convolutional block 5
            block_tensor = layers.Conv2D(128, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block5_1')(block_tensor)

            block_tensor = layers.Conv2D(256, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block5_2')(block_tensor)

            # Classifier + Localizer 6
            #priors_per_cell = 4
            priors_per_cell = 5
            output6_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output6')(block_tensor)

            anchor_shapes.append(self.compute_anchor_shape(priors_per_cell, output1_tensor))
            self.anchors.append(create_anchor_boxes(anchor_shapes[-1], self.aspect_ratios, scale6, self.img_size))

            num_anchors_in_feature_map = SSD300.mult(anchor_shapes[-1][:-1])
            self.outputs.append(Reshape((num_anchors_in_feature_map, self.classes + 4))(output6_tensor))

    def build_layers(self, vgg_output_tensor):
        self.add_ssd300_layers(vgg_output_tensor)
        return Confidence(self.classes)(self.outputs), Localization(self.anchors)(self.outputs)

    def __init__(self, img_size=300, channels=3, classes=21, box_corners=4, freeze_base=True, learning_rate=0.001):
        self.img_size = img_size
        self.channels = channels
        self.classes = classes
        self.box_corners = box_corners
        self.aspect_ratios = [1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3]
        self.min_scale = 0.2
        self.max_scale = 0.9
        self.outputs = []
        self.anchors = []

        vgg16_model = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, self.channels))

        if freeze_base:
            # Freeze first layers
            for layer in vgg16_model.layers:
                layer.trainable = False

        vgg_output_tensor = vgg16_model.outputs[0]

        confidence, localization = self.build_layers(vgg_output_tensor)

        self.model = Model(input=vgg16_model.input, output=[confidence, localization])

        losses, loss_weights = ssd300_loss()
        self.model.compile(optimizer=SGD(lr=learning_rate), loss=losses, loss_weights=loss_weights)

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)

    def evaluate(self, x, y, **kwargs):
        return self.model.evaluate(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.evaluate(x, **kwargs)

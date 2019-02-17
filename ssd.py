import numpy as np
import tensorflow as tf
import keras.layers as layers
from keras.models import Model
from keras.applications.vgg16 import VGG16


class SSD300:
    @staticmethod
    def mult(shape):
        features = 1
        for d in shape:
            features = features * int(d)
        return features

    def compute_output_shape(self, num_priors, output_tensor):
        features = SSD300.mult(output_tensor.shape[1:])
        feature_map_width = int(np.sqrt(features / (num_priors * (self.classes + 4))))
        new_shape = (-1, feature_map_width, feature_map_width, num_priors, 4 + self.classes)
        assert(SSD300.mult(new_shape[1:]) == features)
        return new_shape

    def add_ssd300_layers(self, vgg_output_tensor):
        outputs = []
        with tf.variable_scope('ssd_300'):
            # Classifier 1
            #priors_per_cell = 4
            priors_per_cell = 5
            output1_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output1')(vgg_output_tensor)

            self.output_shapes.append(self.compute_output_shape(priors_per_cell, output1_tensor))

            outputs.append(output1_tensor)

            # Convolutional block 1
            block_tensor = layers.Conv2D(1024, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block1_1')(vgg_output_tensor)

            block_tensor = layers.Conv2D(1024, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block1_2')(block_tensor)

            # Classifier 2
            #priors_per_cell = 6
            priors_per_cell = 5
            output2_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output2')(block_tensor)

            self.output_shapes.append(self.compute_output_shape(priors_per_cell, output2_tensor))

            outputs.append(output2_tensor)

            # Convolutional block 2
            block_tensor = layers.Conv2D(256, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block2_1')(block_tensor)

            block_tensor = layers.Conv2D(512, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block2_2')(block_tensor)

            # Classifier 3
            #priors_per_cell = 6
            priors_per_cell = 5
            output3_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output3')(block_tensor)

            self.output_shapes.append(self.compute_output_shape(priors_per_cell, output3_tensor))

            outputs.append(output3_tensor)

            # Convolutional block 3
            block_tensor = layers.Conv2D(128, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block3_1')(block_tensor)

            block_tensor = layers.Conv2D(256, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block3_2')(block_tensor)

            # Classifier 4
            #priors_per_cell = 6
            priors_per_cell = 5
            output4_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output4')(block_tensor)

            self.output_shapes.append(self.compute_output_shape(priors_per_cell, output4_tensor))

            outputs.append(output4_tensor)

            # Convolutional block 4
            block_tensor = layers.Conv2D(128, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block4_1')(block_tensor)

            block_tensor = layers.Conv2D(256, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block4_2')(block_tensor)

            # Classifier 5
            #priors_per_cell = 4
            priors_per_cell = 5
            output5_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output5')(block_tensor)

            self.output_shapes.append(self.compute_output_shape(priors_per_cell, output5_tensor))

            outputs.append(output5_tensor)

            # Convolutional block 5
            block_tensor = layers.Conv2D(128, (1, 1),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block5_1')(block_tensor)

            block_tensor = layers.Conv2D(256, (3, 3),
                                         activation='relu',
                                         padding='same',
                                         name='ssd_block5_2')(block_tensor)

            # Classifier 6
            #priors_per_cell = 4
            priors_per_cell = 5
            output6_tensor = layers.Conv2D(self.box_corners * priors_per_cell * (self.classes + 4), (3, 3),
                                           activation=None,
                                           padding='same',
                                           name='output6')(block_tensor)

            self.output_shapes.append(self.compute_output_shape(priors_per_cell, output6_tensor))

            outputs.append(output6_tensor)

            return outputs

    def __init__(self, img_size=300, channels=3, classes=21, box_corners=4, freeze_base=True):
        self.img_size = img_size
        self.channels = channels
        self.classes = classes
        self.box_corners = box_corners
        self.output_shapes = []

        vgg16_model = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, self.channels))

        if freeze_base:
            # Freeze first layers
            for layer in vgg16_model.layers:
                layer.trainable = False

        vgg_output_tensor = vgg16_model.outputs[0]

        outputs = self.add_ssd300_layers(vgg_output_tensor)

        self.model = Model(input=vgg16_model.input, output=outputs)

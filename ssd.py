import tensorflow as tf
import keras.layers as layers
from keras.models import Model
from keras.applications.vgg16 import VGG16


BATCH_SIZE = 12
IMG_SIZE = 224
CHANNELS = 3
CLASSES = 20
BOX_CORNERS = 4


class SSD300:
    def __init__(self):
        vgg16_model = VGG16(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

        # Freeze first layers
        for layer in vgg16_model.layers:
            layer.trainable = False

        vgg_output_tensor = vgg16_model.outputs[0]

        outputs = []

        with tf.variable_scope('ssd_300'):
            # Classifier 1
            priors_per_cell = 4
            classifier1_tensor = layers.Conv2D(BOX_CORNERS * priors_per_cell * (CLASSES + 4), (3, 3),
                              activation='softmax',
                              padding='same',
                              name='classifier1')(vgg_output_tensor)

            outputs.append(classifier1_tensor)

            # Convolutional block 1
            output_tensor = layers.Conv2D(1024, (3, 3),
                              activation='relu',
                              padding='same',
                              name='ssd_block1_1')(vgg_output_tensor)

            output_tensor = layers.Conv2D(1024, (1, 1),
                              activation='relu',
                              padding='same',
                              name='ssd_block1_2')(output_tensor)

            # Classifier 2
            priors_per_cell = 6
            classifier2_tensor = layers.Conv2D(BOX_CORNERS * priors_per_cell * (CLASSES + 4), (3, 3),
                              activation='softmax',
                              padding='same',
                              name='classifier2')(output_tensor)

            outputs.append(classifier2_tensor)

            # Convolutional block 2
            output_tensor = layers.Conv2D(256, (1, 1),
                              activation='relu',
                              padding='same',
                              name='ssd_block2_1')(output_tensor)

            output_tensor = layers.Conv2D(512, (3, 3),
                              activation='relu',
                              padding='same',
                              name='ssd_block2_2')(output_tensor)

            # Classifier 3
            priors_per_cell = 6
            classifier3_tensor = layers.Conv2D(BOX_CORNERS * priors_per_cell * (CLASSES + 4), (3, 3),
                              activation='softmax',
                              padding='same',
                              name='classifier3')(output_tensor)

            outputs.append(classifier3_tensor)

            # Convolutional block 3
            output_tensor = layers.Conv2D(128, (1, 1),
                              activation='relu',
                              padding='same',
                              name='ssd_block3_1')(output_tensor)

            # Classifier 4
            priors_per_cell = 6
            classifier4_tensor = layers.Conv2D(BOX_CORNERS * priors_per_cell * (CLASSES + 4), (3, 3),
                              activation='softmax',
                              padding='same',
                              name='classifier4')(output_tensor)

            outputs.append(classifier4_tensor)

            # Convolutional block 4
            output_tensor = layers.Conv2D(128, (1, 1),
                              activation='relu',
                              padding='same',
                              name='ssd_block4_1')(output_tensor)

            # Classifier 5
            priors_per_cell = 4
            classifier5_tensor = layers.Conv2D(BOX_CORNERS * priors_per_cell * (CLASSES + 4), (3, 3),
                              activation='softmax',
                              padding='same',
                              name='classifier5')(output_tensor)

            outputs.append(classifier5_tensor)

            # Convolutional block 5
            output_tensor = layers.Conv2D(128, (1, 1),
                              activation='relu',
                              padding='same',
                              name='ssd_block5_1')(output_tensor)

            # Classifier 6
            priors_per_cell = 4
            classifier6_tensor = layers.Conv2D(BOX_CORNERS * priors_per_cell * (CLASSES + 4), (3, 3),
                              activation='softmax',
                              padding='same',
                              name='classifier6')(output_tensor)

            outputs.append(classifier6_tensor)

        self.model = Model(input=vgg16_model.input, output=outputs)

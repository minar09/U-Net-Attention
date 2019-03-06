from __future__ import print_function
import utils

import tensorflow as tf

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_SET = "10k"
# DATA_SET = "CFPD"
# DATA_SET = "LIP"

L2_SCALE = 1e-4
NUM_OF_CLASSES = 18

if DATA_SET == "CFPD":
    NUM_OF_CLASSES = 23  # Fashion parsing 23 # CFPD

if DATA_SET == "LIP":
    NUM_OF_CLASSES = 20  # human parsing # LIP

"""
  U-NET-PLUS-PLUS
"""


def u_net_plus_plus(inputs, keep_probability, is_training=False):
    net = {}
    l2_reg = L2_SCALE

    # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter

    with tf.variable_scope("inference"):

        # 1, 1, 3
        conv1_1 = utils.conv(
            inputs,
            filters=64,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv1_2 = utils.conv(
            conv1_1,
            filters=64,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool1 = utils.pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = utils.conv(
            pool1,
            filters=128,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv2_2 = utils.conv(
            conv2_1,
            filters=128,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool2 = utils.pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = utils.conv(
            pool2,
            filters=256,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv3_2 = utils.conv(
            conv3_1,
            filters=256,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool3 = utils.pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = utils.conv(
            pool3,
            filters=512,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        conv4_2 = utils.conv(
            conv4_1,
            filters=512,
            l2_reg_scale=l2_reg,
            batchnorm_istraining=is_training)
        pool4 = utils.pool(conv4_2)

        # 1/16, 1/16, 512
        conv5_1 = utils.conv(pool4, filters=1024, l2_reg_scale=l2_reg)
        conv5_2 = utils.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
        concated1 = tf.concat([utils.conv_transpose(
            conv5_2, filters=512, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = utils.conv(concated1, filters=512, l2_reg_scale=l2_reg)
        conv_up1_2 = utils.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
        concated2 = tf.concat([utils.conv_transpose(
            conv_up1_2, filters=256, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = utils.conv(concated2, filters=256, l2_reg_scale=l2_reg)
        conv_up2_2 = utils.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
        concated3 = tf.concat([utils.conv_transpose(
            conv_up2_2, filters=128, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = utils.conv(concated3, filters=128, l2_reg_scale=l2_reg)
        conv_up3_2 = utils.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
        concated4 = tf.concat([utils.conv_transpose(
            conv_up3_2, filters=64, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = utils.conv(concated4, filters=64, l2_reg_scale=l2_reg)
        conv_up4_2 = utils.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg)
        outputs = utils.conv(
            conv_up4_2, filters=NUM_OF_CLASSES, kernel_size=[
                1, 1], activation=None)
        annotation_pred = tf.argmax(outputs, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), outputs, net
        # return Model(inputs, outputs, teacher, is_training)

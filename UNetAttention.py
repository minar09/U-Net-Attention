from __future__ import print_function

import BatchDatsetReader as DataSetReader
import Read10kData as fashion_parsing
import ReadCFPDdata as ClothingParsing
import ReadLIPdata as HumanParsing
import TensorflowUtils as Utils
import FunctionDefinitions as fd

import tensorflow as tf

# Hide the warning messages about CPU/GPU
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_SET = "10k"
# DATA_SET = "CFPD"
# DATA_SET = "LIP"

FLAGS = tf.flags.FLAGS

if DATA_SET == "10k":
    tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
    tf.flags.DEFINE_integer(
        "training_epochs",
        "50",
        "number of epochs for training")
    tf.flags.DEFINE_string("logs_dir", "logs/UNetAttention_10k/",
                           "path to logs directory")
    tf.flags.DEFINE_string(
        "data_dir", "D:/Datasets/Dressup10k/", "path to dataset")

if DATA_SET == "CFPD":
    tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
    tf.flags.DEFINE_integer(
        "training_epochs",
        "70",
        "number of epochs for training")
    tf.flags.DEFINE_string("logs_dir", "logs/UNetAttention_CFPD/",
                           "path to logs directory")
    tf.flags.DEFINE_string(
        "data_dir", "D:/Datasets/CFPD/", "path to dataset")

if DATA_SET == "LIP":
    tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
    tf.flags.DEFINE_integer(
        "training_epochs",
        "30",
        "number of epochs for training")
    tf.flags.DEFINE_string("logs_dir", "logs/UNetAttention_LIP/",
                           "path to logs directory")
    tf.flags.DEFINE_string(
        "data_dir", "D:/Datasets/LIP/", "path to dataset")

tf.flags.DEFINE_float(
    "learning_rate",
    "1e-4",
    "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")

tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
# tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")
# tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1001)

NUM_OF_CLASSES = 18  # Upper-lower cloth parsing # Dressup 10k
DISPLAY_STEP = 300

if DATA_SET == "CFPD":
    NUM_OF_CLASSES = 23  # Fashion parsing 23 # CFPD

if DATA_SET == "LIP":
    NUM_OF_CLASSES = 20  # human parsing # LIP

# IMAGE_SIZE = 224
IMAGE_SIZE = 384
TEST_DIR = FLAGS.logs_dir + "TestImage/"
VIS_DIR = FLAGS.logs_dir + "VisImage/"


"""
  UNET
"""


def u_net_inference(image, is_training=False):
    net = {}
    l2_reg = FLAGS.learning_rate

    # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter

    with tf.variable_scope("inference"):
        inputs = image

        # 1, 1, 3 Encoder 1st
        conv1_1 = Utils.conv(
            inputs,
            filters=64,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        conv1_2 = Utils.conv(
            conv1_1,
            filters=64,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        pool1 = Utils.pool(conv1_2)

        # 1/2, 1/2, 64 Encoder 2nd
        conv2_1 = Utils.conv(
            pool1,
            filters=128,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        conv2_2 = Utils.conv(
            conv2_1,
            filters=128,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        pool2 = Utils.pool(conv2_2)

        # 1/4, 1/4, 128 Encoder 3rd
        conv3_1 = Utils.conv(
            pool2,
            filters=256,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        conv3_2 = Utils.conv(
            conv3_1,
            filters=256,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        pool3 = Utils.pool(conv3_2)

        # 1/8, 1/8, 256 Encoder 4th
        conv4_1 = Utils.conv(
            pool3,
            filters=512,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        conv4_2 = Utils.conv(
            conv4_1,
            filters=512,
            l2_reg_scale=l2_reg,
            is_training=is_training)
        pool4 = Utils.pool(conv4_2)

        # 1/16, 1/16, 512 Encoder 5th, upsample, skip 1
        conv5_1 = Utils.conv(pool4, filters=1024, l2_reg_scale=l2_reg, is_training=is_training)
        conv5_2 = Utils.conv(conv5_1, filters=1024, l2_reg_scale=l2_reg, is_training=is_training)
        concated1 = tf.concat([Utils.conv_transpose(
            conv5_2, filters=512, l2_reg_scale=l2_reg, is_training=is_training), conv4_2], axis=3)

        # Decoder 1st, skip 2
        conv_up1_1 = Utils.conv(concated1, filters=512, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up1_2 = Utils.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg, is_training=is_training)
        concated2 = tf.concat([Utils.conv_transpose(
            conv_up1_2, filters=256, l2_reg_scale=l2_reg, is_training=is_training), conv3_2], axis=3)

        # Decoder 2nd, skip 3
        conv_up2_1 = Utils.conv(concated2, filters=256, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up2_2 = Utils.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg, is_training=is_training)
        concated3 = tf.concat([Utils.conv_transpose(
            conv_up2_2, filters=128, l2_reg_scale=l2_reg, is_training=is_training), conv2_2], axis=3)

        # Decoder 3rd, skip 4
        conv_up3_1 = Utils.conv(concated3, filters=128, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up3_2 = Utils.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg, is_training=is_training)
        concated4 = tf.concat([Utils.conv_transpose(
            conv_up3_2, filters=64, l2_reg_scale=l2_reg, is_training=is_training), conv1_2], axis=3)

        # Decoder 4th
        conv_up4_1 = Utils.conv(concated4, filters=64, l2_reg_scale=l2_reg, is_training=is_training)
        conv_up4_2 = Utils.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg, is_training=is_training)
        # logits/probability
        logits = Utils.conv(
            conv_up4_2, filters=NUM_OF_CLASSES, kernel_size=[
                1, 1], activation=None)
        # Output/prediction
        annotation_pred = tf.argmax(logits, dimension=3, name="prediction")
        outputs = tf.expand_dims(annotation_pred, dim=3)

        return outputs, logits, net, conv5_2


"""
    Attention model
"""


def attention(scale_input, is_training=False):
    l2_reg = FLAGS.learning_rate
    dropout_ratio = 0
    if is_training is True:
        dropout_ratio = 0.5

    conv1 = Utils.conv(scale_input, filters=512, l2_reg_scale=l2_reg, is_training=is_training)
    conv1 = Utils.dropout(conv1, dropout_ratio, is_training)
    conv2 = Utils.conv(conv1, filters=3, kernel_size=[1, 1], l2_reg_scale=l2_reg, is_training=is_training)
    return conv2


"""inference
  optimize with trainable parameters (Check which ones)
  loss_val : loss operator (mean(

"""


def train(loss_val, var_list, global_step):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            Utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


def main(argv=None):
    # 1. input placeholders
    image = tf.placeholder(
        tf.float32,
        shape=(
            None,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3),
        name="input_image")
    annotation = tf.placeholder(
        tf.int32,
        shape=(
            None,
            IMAGE_SIZE,
            IMAGE_SIZE,
            1),
        name="annotation")
    training = tf.placeholder(
        tf.bool,
        shape=None,
        name="is_training")

    # global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = False
    if FLAGS.mode == "train":
        is_training = True

    image075 = tf.image.resize_images(
        image, [int(IMAGE_SIZE * 0.75), int(IMAGE_SIZE * 0.75)])
    image050 = tf.image.resize_images(
        image, [int(IMAGE_SIZE * 0.5), int(IMAGE_SIZE * 0.5)])
    image125 = tf.image.resize_images(
        image, [int(IMAGE_SIZE * 1.25), int(IMAGE_SIZE * 1.25)])

    annotation075 = tf.cast(tf.image.resize_images(
        annotation, [int(IMAGE_SIZE * 0.75), int(IMAGE_SIZE * 0.75)]), tf.int32)
    annotation050 = tf.cast(tf.image.resize_images(
        annotation, [int(IMAGE_SIZE * 0.5), int(IMAGE_SIZE * 0.5)]), tf.int32)
    annotation125 = tf.cast(tf.image.resize_images(
        annotation, [int(IMAGE_SIZE * 1.25), int(IMAGE_SIZE * 1.25)]), tf.int32)

    # 2. construct inference network
    reuse1 = False
    reuse2 = True  # For sharing weights among the latter scales

    with tf.variable_scope('', reuse=reuse1):
        pred_annotation100, logits100, net100, att100 = u_net_inference(image, is_training=is_training)
    with tf.variable_scope('', reuse=reuse2):
        pred_annotation075, logits075, net075, att075 = u_net_inference(image075, is_training=is_training)
    with tf.variable_scope('', reuse=reuse2):
        pred_annotation050, logits050, net050, att050 = u_net_inference(image050, is_training=is_training)
    with tf.variable_scope('', reuse=reuse2):
        pred_annotation125, logits125, net125, att125 = u_net_inference(image125, is_training=is_training)

    # apply attention model - train
    score_final_test = None
    final_annotation_pred_test = None

    train_op = None
    reduced_loss = None

    if FLAGS.mode == "train":
        attn_input = []
        attn_input.append(att100)
        attn_input.append(tf.image.resize_images(att075, tf.shape(att100)[1:3, ]))
        attn_input.append(tf.image.resize_images(att050, tf.shape(att100)[1:3, ]))
        attn_input_train = tf.concat(attn_input, axis=3)
        attn_output_train = attention(attn_input_train, is_training)
        scale_att_mask = tf.nn.softmax(attn_output_train, axis=3)    # Add axis?

        score_att_x = tf.multiply(logits100, tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 0], axis=3), tf.shape(logits100)[1:3, ]))
        score_att_x_075 = tf.multiply(tf.image.resize_images(logits075, tf.shape(logits100)[1:3, ]), tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 1], axis=3), tf.shape(logits100)[1:3, ]))
        score_att_x_050 = tf.multiply(tf.image.resize_images(logits050, tf.shape(logits100)[1:3, ]), tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 2], axis=3), tf.shape(logits100)[1:3, ]))
        score_final_train = score_att_x + score_att_x_075 + score_att_x_050

        final_annotation_pred = tf.expand_dims(tf.argmax(score_final_train, dimension=3, name="final_prediction"), dim=3)
        final_annotation_pred_train = tf.reduce_mean(
            tf.stack([tf.cast(final_annotation_pred, tf.float32), tf.cast(pred_annotation100, tf.float32),
                      tf.image.resize_images(pred_annotation075,
                                             tf.shape(pred_annotation100)[1:3, ]),
                      tf.image.resize_images(pred_annotation050,
                                             tf.shape(pred_annotation100)[1:3, ])]),
            axis=0)
        # final_annotation_pred_train = tf.expand_dims(tf.argmax(final_annotation_pred_train, dimension=3, name="final_prediction"), dim=3)

        # 3. loss measure
        loss = tf.reduce_mean(
            (tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=score_final_train,
                labels=tf.squeeze(
                    annotation,
                    squeeze_dims=[3]),
                name="entropy")))
        tf.summary.scalar("entropy", loss)

        loss100 = tf.reduce_mean(
            (tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits100,
                labels=tf.squeeze(
                    annotation,
                    squeeze_dims=[3]),
                name="entropy")))
        tf.summary.scalar("entropy", loss100)

        loss075 = tf.reduce_mean(
            (tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits075,
                labels=tf.squeeze(
                    annotation075,
                    squeeze_dims=[3]),
                name="entropy")))
        tf.summary.scalar("entropy", loss075)

        loss050 = tf.reduce_mean(
            (tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits050,
                labels=tf.squeeze(
                    annotation050,
                    squeeze_dims=[3]),
                name="entropy")))
        tf.summary.scalar("entropy", loss050)

        reduced_loss = loss + loss100 + loss075 + loss050

        # 4. optimizing
        trainable_var = tf.trainable_variables()
        if FLAGS.debug:
            for var in trainable_var:
                Utils.add_to_regularization_and_summary(var)

        train_op = train(reduced_loss, trainable_var, net100['global_step'])

    else:
        # apply attention model - test
        attn_input = []
        attn_input.append(att100)
        attn_input.append(tf.image.resize_images(att075, tf.shape(att100)[1:3, ]))
        attn_input.append(tf.image.resize_images(att125, tf.shape(att100)[1:3, ]))
        attn_input_test = tf.concat(attn_input, axis=3)
        attn_output_test = attention(attn_input_test, is_training)
        scale_att_mask = tf.nn.softmax(attn_output_test, axis=3)    # Add axis?

        score_att_x = tf.multiply(logits100, tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 0], axis=3),
                                                                    tf.shape(logits100)[1:3, ]))
        score_att_x_075 = tf.multiply(tf.image.resize_images(logits075, tf.shape(logits100)[1:3, ]),
                                      tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 1], axis=3),
                                                             tf.shape(logits100)[1:3, ]))
        score_att_x_125 = tf.multiply(tf.image.resize_images(logits125, tf.shape(logits100)[1:3, ]),
                                      tf.image.resize_images(tf.expand_dims(scale_att_mask[:, :, :, 2], axis=3),
                                                             tf.shape(logits100)[1:3, ]))

        score_final_test = score_att_x + score_att_x_075 + score_att_x_125

        final_annotation_pred = tf.expand_dims(tf.argmax(score_final_test, dimension=3, name="final_prediction"), dim=3)
        final_annotation_pred_test = tf.reduce_mean(tf.stack([tf.cast(final_annotation_pred, tf.float32), tf.cast(pred_annotation100, tf.float32),
                                                        tf.image.resize_images(pred_annotation075,
                                                                               tf.shape(pred_annotation100)[1:3, ]),
                                                        tf.image.resize_images(pred_annotation125,
                                                                               tf.shape(pred_annotation100)[1:3, ])]),
                                              axis=0)
        # final_annotation_pred_test = tf.expand_dims(tf.argmax(final_annotation_pred_test, dimension=3, name="final_prediction"), dim=3)

    tf.summary.image("input_image", image, max_outputs=3)
    tf.summary.image(
        "ground_truth",
        tf.cast(
            annotation,
            tf.uint8),
        max_outputs=3)

    tf.summary.image(
        "pred_annotation",
        tf.cast(
            pred_annotation100,
            tf.uint8),
        max_outputs=3)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader from ", FLAGS.data_dir, "...")
    print("data dir:", FLAGS.data_dir)

    train_records, valid_records = fashion_parsing.read_dataset(FLAGS.data_dir)
    test_records = None
    if DATA_SET == "CFPD":
        train_records, valid_records, test_records = ClothingParsing.read_dataset(
            FLAGS.data_dir)
        print("test_records length :", len(test_records))
    if DATA_SET == "LIP":
        train_records, valid_records = HumanParsing.read_dataset(
            FLAGS.data_dir)

    print("train_records length :", len(train_records))
    print("valid_records length :", len(valid_records))

    print("Setting up dataset reader")
    train_dataset_reader = None
    validation_dataset_reader = None
    test_dataset_reader = None
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}

    if FLAGS.mode == 'train':
        train_dataset_reader = DataSetReader.BatchDatset(
            train_records, image_options)
        validation_dataset_reader = DataSetReader.BatchDatset(
            valid_records, image_options)
        if DATA_SET == "CFPD":
            test_dataset_reader = DataSetReader.BatchDatset(
                test_records, image_options)
    if FLAGS.mode == 'visualize':
        validation_dataset_reader = DataSetReader.BatchDatset(
            valid_records, image_options)
    if FLAGS.mode == 'test' or FLAGS.mode == 'crftest' or FLAGS.mode == 'predonly' or FLAGS.mode == "fulltest":
        if DATA_SET == "CFPD":
            test_dataset_reader = DataSetReader.BatchDatset(
                test_records, image_options)
        else:
            test_dataset_reader = DataSetReader.BatchDatset(
                valid_records, image_options)
            test_records = valid_records

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    # 5. parameter setup
    # 5.1 init params
    sess.run(tf.global_variables_initializer())
    # 5.2 restore params if possible
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # 6. train-mode
    if FLAGS.mode == "train":

        fd.mode_train(sess, FLAGS, net100, train_dataset_reader, validation_dataset_reader, final_annotation_pred,
                      image, annotation, training, train_op, reduced_loss, summary_op, summary_writer,
                      saver)

    # test-random-validation-data mode
    elif FLAGS.mode == "visualize":

        fd.mode_visualize_scales(sess, FLAGS, VIS_DIR, validation_dataset_reader,
                                 scale_att_mask, final_annotation_pred, pred_annotation100, pred_annotation075, pred_annotation125, image, annotation, training, NUM_OF_CLASSES)

    # test-full-validation-dataset mode
    elif FLAGS.mode == "test":

        fd.mode_test(sess, FLAGS, TEST_DIR, test_dataset_reader,
                     final_annotation_pred, image, annotation, training, score_final_test, NUM_OF_CLASSES)

    sess.close()


if __name__ == "__main__":
    tf.app.run()

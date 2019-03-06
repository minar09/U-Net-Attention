from __future__ import print_function

import utils
import time
import matplotlib.pyplot as plt
from six.moves import xrange
import datetime
import tensorflow as tf
import numpy as np
import BatchDatsetReader
import ReadDataset
import model

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


DATA_SET = "10k"
# DATA_SET = "CFPD"
# DATA_SET = "LIP"

BATCH_SIZE = 40
NUM_EPOCHS = 50
LOG_DIR = "logs/10k/"
DATA_DIR = "D:/Datasets/Dressup10k/"
LEARNING_RATE = 1e-4
DEBUG = False
NUM_OF_CLASSES = 18  # Upper-lower cloth parsing # Dressup 10k
DISPLAY_STEP = 300
MAX_ITERATION = int(1e5 + 1001)
IMAGE_SIZE = 224
TEST_DIR = LOG_DIR + "Evaluation/"
VIS_DIR = LOG_DIR + "Visualization/"

if DATA_SET == "CFPD":
    BATCH_SIZE = 38
    NUM_EPOCHS = 70
    LOG_DIR = "logs/CFPD/"
    DATA_DIR = "D:/Datasets/CFPD/"
    NUM_OF_CLASSES = 23  # Fashion parsing 23 # CFPD

if DATA_SET == "LIP":
    BATCH_SIZE = 40
    NUM_EPOCHS = 30
    LOG_DIR = "logs/LIP/"
    DATA_DIR = "D:/Datasets/LIP/"
    NUM_OF_CLASSES = 20  # human parsing # LIP


def train(loss_val, var_list, global_step):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if DEBUG:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step=global_step)


def main():
    # input placeholders
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
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
    # global_step = tf.Variable(0, trainable=False, name='global_step')

    # 2. construct inference network
    pred_annotation, logits, net = model.u_net_plus_plus(image, keep_probability, is_training=True)
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
            pred_annotation,
            tf.uint8),
        max_outputs=3)

    # 3. loss measure
    loss = tf.reduce_mean(
        (tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.squeeze(
                annotation,
                squeeze_dims=[3]),
            name="entropy")))
    tf.summary.scalar("entropy", loss)

    # 4. optimizing
    trainable_var = tf.trainable_variables()
    if DEBUG:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    train_op = train(loss, trainable_var, net['global_step'])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader from ", DATA_DIR, "...")
    print("data dir:", DATA_DIR)

    train_records, valid_records = ReadDataset.read_dataset(DATA_DIR, DATA_SET)
    test_records = None
    if DATA_SET == "CFPD":
        train_records, valid_records, test_records = ReadDataset.read_dataset(
            DATA_DIR, DATA_SET)
        print("test_records length :", len(test_records))
    if DATA_SET == "LIP":
        train_records, valid_records = ReadDataset.read_dataset(
            DATA_DIR, DATA_SET)

    print("train_records length :", len(train_records))
    print("valid_records length :", len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    train_dataset_reader = BatchDatsetReader.BatchDatset(
        train_records, image_options)
    validation_dataset_reader = BatchDatsetReader.BatchDatset(
        valid_records, image_options)
    if DATA_SET == "CFPD":
        test_dataset_reader = BatchDatsetReader.BatchDatset(
            test_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # 5. parameter setup
    # 5.1 init params
    sess.run(tf.global_variables_initializer())
    # 5.2 restore params if possible
    ckpt = tf.train.get_checkpoint_state(LOG_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # 6. train-mode
    print(">>>>>>>>>>>>>>>>Train mode")
    start = time.time()

    # Start decoder training

    valid = list()
    step = list()
    losses = list()

    global_step = sess.run(net['global_step'])
    max_iteration = round(
        (train_dataset_reader.get_num_of_records() //
         BATCH_SIZE) *
        NUM_EPOCHS)
    display_step = round(
        train_dataset_reader.get_num_of_records() // BATCH_SIZE)
    print(
        "No. of maximum steps:",
        max_iteration,
        " Training epochs:",
        NUM_EPOCHS)

    for itr in xrange(global_step, max_iteration):
        # 6.1 load train and GT images
        train_images, train_annotations = train_dataset_reader.next_batch(
            BATCH_SIZE)

        feed_dict = {
            image: train_images,
            annotation: train_annotations,
            keep_probability: 0.85}

        # 6.2 training
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 10 == 0:
            train_loss, summary_str = sess.run(
                [loss, summary_op], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            summary_writer.add_summary(summary_str, itr)
            if itr % display_step == 0 and itr != 0:
                losses.append(train_loss)

        if itr % display_step == 0 and itr != 0:
            valid_images, valid_annotations = validation_dataset_reader.next_batch(
                BATCH_SIZE)
            valid_loss = sess.run(
                loss,
                feed_dict={
                    image: valid_images,
                    annotation: valid_annotations,
                    keep_probability: 1.0})
            print(
                "%s ---> Validation_loss: %g" %
                (datetime.datetime.now(), valid_loss))
            global_step = sess.run(net['global_step'])
            saver.save(
                sess,
                LOG_DIR +
                "model.ckpt",
                global_step=global_step)

            valid.append(valid_loss)
            step.append(itr)
            # print("valid", valid, "step", step)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(losses))
                plt.title('Training Loss')
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.savefig(LOG_DIR + "training_loss.jpg")
            except Exception as err:
                print(err)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(valid))
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Validation Loss')
                plt.savefig(LOG_DIR + "validation_loss.jpg")
            except Exception as err:
                print(err)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(losses))
                plt.plot(np.array(step), np.array(valid))
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Result')
                plt.legend(['Training Loss', 'Validation Loss'],
                           loc='upper right')
                plt.savefig(LOG_DIR + "merged_loss.jpg")
            except Exception as err:
                print(err)

    try:
        np.savetxt(
            LOG_DIR +
            "training_steps.csv",
            np.c_[step, losses, valid],
            fmt='%4f',
            delimiter=',')
    except Exception as err:
        print(err)

    end = time.time()
    print("Learning time:", end - start, "seconds")

    sess.close()


if __name__ == "__main__":
    main()

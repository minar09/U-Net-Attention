from __future__ import print_function

import utils
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import BatchDatsetReader
import ReadDataset
import model
import EvalMetrics
import ApplyCRF

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
    pred_annotation, logits, net = model.u_net_plus_plus(image, keep_probability, is_training=False)
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

    # 6. test-mode
    print(">>>>>>>>>>>>>>>>Test mode")
    start = time.time()

    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    validation_dataset_reader.reset_batch_offset(0)

    crossMats = list()
    crf_crossMats = list()

    for itr1 in range(validation_dataset_reader.get_num_of_records() // BATCH_SIZE):

        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            BATCH_SIZE)
        pred, logits1 = sess.run([pred_annotation, logits],
                                 feed_dict={image: valid_images, annotation: valid_annotations,
                                            keep_probability: 1.0})

        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred)
        print("logits shape:", logits1.shape)
        np.set_printoptions(threshold=np.inf)

        for itr2 in range(BATCH_SIZE):
            fig = plt.figure()
            pos = 240 + 1
            plt.subplot(pos)
            plt.imshow(valid_images[itr2].astype(np.uint8))
            plt.axis('off')
            plt.title('Original')

            pos = 240 + 2
            plt.subplot(pos)
            plt.imshow(
                valid_annotations[itr2].astype(
                    np.uint8),
                cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('GT')

            pos = 240 + 3
            plt.subplot(pos)
            plt.imshow(
                pred[itr2].astype(
                    np.uint8),
                cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('Prediction')

            # Confusion matrix for this image prediction
            crossMat = EvalMetrics._calcCrossMat(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), NUM_OF_CLASSES)
            crossMats.append(crossMat)

            np.savetxt(TEST_DIR +
                       "Crossmatrix" +
                       str(itr1 *
                           BATCH_SIZE +
                           itr2) +
                       ".csv", crossMat, fmt='%4i', delimiter=',')

            # Save input, gt, pred, crf_pred, sum figures for this image

            # ---------------------------------------------
            utils.save_image(valid_images[itr2].astype(np.uint8), TEST_DIR,
                             name="inp_" + str(itr1 * BATCH_SIZE + itr2))
            utils.save_image(valid_annotations[itr2].astype(np.uint8), TEST_DIR,
                             name="gt_" + str(itr1 * BATCH_SIZE + itr2))
            utils.save_image(pred[itr2].astype(np.uint8),
                             TEST_DIR,
                             name="pred_" + str(itr1 * BATCH_SIZE + itr2))

            # --------------------------------------------------
            """ Generate CRF """
            crfimage, crfoutput = ApplyCRF.crf(TEST_DIR + "inp_" + str(itr1 * BATCH_SIZE + itr2) + ".png",
                                                TEST_DIR + "pred_" + str(
                                                    itr1 * BATCH_SIZE + itr2) + ".png",
                                                TEST_DIR + "crf_" + str(itr1 * BATCH_SIZE + itr2) + ".png",
                                                NUM_OF_CLASSES, use_2d=True)

            # Confusion matrix for this image prediction with crf
            crf_crossMat = EvalMetrics._calcCrossMat(
                valid_annotations[itr2].astype(
                    np.uint8), crfoutput.astype(
                    np.uint8), NUM_OF_CLASSES)
            crf_crossMats.append(crf_crossMat)

            np.savetxt(TEST_DIR +
                       "crf_Crossmatrix" +
                       str(itr1 *
                           BATCH_SIZE +
                           itr2) +
                       ".csv", crf_crossMat, fmt='%4i', delimiter=',')

            pos = 240 + 4
            plt.subplot(pos)
            plt.imshow(crfoutput.astype(np.uint8),
                       cmap=plt.get_cmap('nipy_spectral'))
            plt.axis('off')
            plt.title('Prediction + CRF')

            plt.savefig(TEST_DIR + "resultSum_" +
                        str(itr1 * BATCH_SIZE + itr2))

            plt.close('all')
            print("Saved image: %d" % (itr1 * BATCH_SIZE + itr2))

    try:
        total_cm = np.sum(crossMats, axis=0)
        np.savetxt(
            LOG_DIR +
            "Crossmatrix.csv",
            total_cm,
            fmt='%4i',
            delimiter=',')

        print(">>> Prediction results:")
        EvalMetrics.show_result(total_cm, NUM_OF_CLASSES)

        # Prediction with CRF
        crf_total_cm = np.sum(crf_crossMats, axis=0)
        np.savetxt(
            LOG_DIR +
            "CRF_Crossmatrix.csv",
            crf_total_cm,
            fmt='%4i',
            delimiter=',')

        print("\n")
        print(">>> Prediction results (CRF):")
        EvalMetrics.show_result(crf_total_cm, NUM_OF_CLASSES)

    except Exception as err:
        print(err)

    end = time.time()
    print("Testing time:", end - start, "seconds")

    sess.close()


if __name__ == "__main__":
    main()

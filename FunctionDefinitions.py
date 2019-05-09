from __future__ import print_function

import matplotlib.pyplot as plt
from six.moves import xrange
import datetime

import numpy as np
import tensorflow as tf
import time
import EvalMetrics
import denseCRF
import TensorflowUtils as Utils
from matplotlib.colors import ListedColormap, BoundaryNorm

# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


label_colors_10k = ['black',  # "background", #     0
                'sienna',  # "hat", #            1
                'gray',  # "hair", #           2
                'navy',  # "sunglass", #       3
                'red',  # "upper-clothes", #  4
                'gold',  # "skirt",  #          5
                'blue',  # "pants",  #          6
                'seagreen',  # "dress", #          7
                'darkorchid',  # "belt", #           8
                'firebrick',  # "left-shoe", #      9
                    'darksalmon',  # "right-shoe", #     10
                    'moccasin',  # "face",  #           11
                    'darkgreen',  # "left-leg", #       12
                    'royalblue',  # "right-leg", #      13
                    'chartreuse',  # "left-arm",#       14
                    'paleturquoise',  # "right-arm", #      15
                    'darkcyan',  # "bag", #            16
                    'deepskyblue'  # "scarf" #          17
                    ]

clothnorm_10k = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
                              7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], 18)

# colour map for LIP dataset
lip_label_colours = [(0, 0, 0),  # 0=Background
                     (128, 0, 0),  # 1=Hat
                     (255, 0, 0),  # 2=Hair
                     (0, 85, 0),  # 3=Glove
                     (170, 0, 51),  # 4=Sunglasses
                     (255, 85, 0),  # 5=UpperClothes
                     (0, 0, 85),  # 6=Dress
                     (0, 119, 221),  # 7=Coat
                     (85, 85, 0),  # 8=Socks
                     (0, 85, 85),  # 9=Pants
                     (85, 51, 0),  # 10=Jumpsuits
                     (52, 86, 128),  # 11=Scarf
                     (0, 128, 0),  # 12=Skirt
                     (0, 0, 255),  # 13=Face
                     (51, 170, 221),  # 14=LeftArm
                     (0, 255, 255),  # 15=RightArm
                     (85, 255, 170),  # 16=LeftLeg
                     (170, 255, 85),  # 17=RightLeg
                     (255, 255, 0),  # 18=LeftShoe
                     (255, 170, 0)  # 19=RightShoe
                     ]


def mode_visualize(sess, flags, test_dir, validation_dataset_reader, attn_output_test, pred_annotation, pred_annotation100, pred_annotation075, pred_annotation125, image, annotation, keep_probability, num_classes):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    valid_images, valid_annotations = validation_dataset_reader.get_random_batch(
        flags.batch_size)
    pred, weights, pred100, pred75, pred125 = sess.run([pred_annotation, attn_output_test, pred_annotation100, pred_annotation075, pred_annotation125],
                                                      feed_dict={image: valid_images, annotation: valid_annotations,
                               keep_probability: 1.0})

    valid_annotations = np.squeeze(valid_annotations, axis=3)
    pred = np.squeeze(pred, axis=3)
    pred100 = np.squeeze(pred100, axis=3)
    pred75 = np.squeeze(pred75, axis=3)
    pred125 = np.squeeze(pred125, axis=3)

    crossMats = list()

    for itr in range(flags.batch_size):
        print("Saved image: %d" % itr)

        # Eval metrics for this image prediction
        cm = EvalMetrics.calculate_confusion_matrix(
            valid_annotations[itr].astype(
                np.uint8), pred[itr].astype(
                np.uint8), num_classes)
        crossMats.append(cm)

        fig = plt.figure()
        pos = 240 + 1
        plt.subplot(pos)
        plt.imshow(valid_images[itr].astype(np.uint8))
        plt.axis('off')
        plt.title('Original')

        pos = 240 + 2
        plt.subplot(pos)
        plt.imshow(
            valid_annotations[itr].astype(
                np.uint8),
                       cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
        plt.axis('off')
        plt.title('GT')

        pos = 240 + 3
        plt.subplot(pos)
        plt.imshow(
            weights[itr].astype(
                np.uint8),
                       cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
        plt.axis('off')
        plt.title('Weights')

        pos = 240 + 4
        plt.subplot(pos)
        plt.imshow(
            pred[itr].astype(
                np.uint8),
                       cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
        plt.axis('off')
        plt.title('Prediction')

        pos = 240 + 5
        plt.subplot(pos)
        plt.imshow(
            pred100[itr].astype(
                np.uint8),
            cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
        plt.axis('off')
        plt.title('Prediction100')

        pos = 240 + 6
        plt.subplot(pos)
        plt.imshow(
            pred75[itr].astype(
                np.uint8),
            cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
        plt.axis('off')
        plt.title('Prediction075')

        pos = 240 + 7
        plt.subplot(pos)
        plt.imshow(
            pred125[itr].astype(
                np.uint8),
            cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
        plt.axis('off')
        plt.title('Prediction125')

        plt.savefig(test_dir + "resultSum_" +
                    str(itr))

        plt.close('all')

    print(">>> Prediction results:")
    total_cm = np.sum(crossMats, axis=0)
    EvalMetrics.show_result(total_cm, num_classes)


def mode_train(sess, FLAGS, net, train_dataset_reader, validation_dataset_reader, image, annotation, keep_probability, train_op, loss, summary_op, summary_writer, saver):
    print(">>>>>>>>>>>>>>>>Train mode")
    start = time.time()

    # Start decoder training

    valid = list()
    step = list()
    lo = list()

    global_step = sess.run(net['global_step'])
    global_step = 0
    max_iteration = round(
        (train_dataset_reader.get_num_of_records() //
         FLAGS.batch_size) *
        FLAGS.training_epochs)
    display_step = round(
        train_dataset_reader.get_num_of_records() // FLAGS.batch_size)
    print(
        "No. of maximum steps:",
        max_iteration,
        " Training epochs:",
        FLAGS.training_epochs)

    for itr in xrange(global_step, max_iteration):
        # 6.1 load train and GT images
        train_images, train_annotations = train_dataset_reader.next_batch(
            FLAGS.batch_size)

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
                lo.append(train_loss)

        if itr % display_step == 0 and itr != 0:
            valid_images, valid_annotations = validation_dataset_reader.next_batch(
                FLAGS.batch_size)
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
                FLAGS.logs_dir +
                "model.ckpt",
                global_step=global_step)

            valid.append(valid_loss)
            step.append(itr)
            # print("valid", valid, "step", step)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(lo))
                plt.title('Training Loss')
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.savefig(FLAGS.logs_dir + "training_loss.jpg")
            except Exception as err:
                print(err)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(valid))
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Validation Loss')
                plt.savefig(FLAGS.logs_dir + "validation_loss.jpg")
            except Exception as err:
                print(err)

            try:
                plt.clf()
                plt.ylim(0, 1)
                plt.plot(np.array(step), np.array(lo))
                plt.plot(np.array(step), np.array(valid))
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Result')
                plt.legend(['Training Loss', 'Validation Loss'],
                           loc='upper right')
                plt.savefig(FLAGS.logs_dir + "merged_loss.jpg")
            except Exception as err:
                print(err)

    try:
        np.savetxt(
            FLAGS.logs_dir +
            "training_steps.csv",
            np.c_[step, lo, valid],
            fmt='%4f',
            delimiter=',')
    except Exception as err:
        print(err)

    end = time.time()
    print("Learning time:", end - start, "seconds")


def mode_test(sess, flags, save_dir, validation_dataset_reader, pred_annotation, image, annotation, keep_probability, logits, num_classes):
    print(">>>>>>>>>>>>>>>>Test mode")
    start = time.time()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    validation_dataset_reader.reset_batch_offset(0)
    probability = tf.nn.softmax(logits=logits, axis=3)

    cross_mats = list()
    crf_cross_mats = list()

    # tf_pixel_acc_list = []
    # tf_miou_list = []

    # pixel_acc_op, pixel_acc_update_op = tf.metrics.accuracy(labels=annotation, predictions=pred_annotation)
    # mean_iou_op, mean_iou_update_op = tf.metrics.mean_iou(labels=annotation, predictions=pred_annotation, num_classes=num_classes)

    for itr1 in range(validation_dataset_reader.get_num_of_records() // flags.batch_size):

        valid_images, valid_annotations = validation_dataset_reader.next_batch(
            flags.batch_size)

        predprob, pred = sess.run([probability, pred_annotation], feed_dict={image: valid_images, keep_probability: 1.0})

        # tf measures
        sess.run(tf.local_variables_initializer())
        feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 1.0}
        # predprob, pred, _, __ = sess.run([probability, pred_annotation, pixel_acc_update_op, mean_iou_update_op], feed_dict=feed_dict)
        # tf_pixel_acc, tf_miou = sess.run([pixel_acc_op, mean_iou_op], feed_dict=feed_dict)
        # tf_pixel_acc_list.append(tf_pixel_acc)
        # tf_miou_list.append(tf_miou)

        np.set_printoptions(threshold=10)

        pred = np.squeeze(pred)
        predprob = np.squeeze(predprob)
        valid_annotations = np.squeeze(valid_annotations, axis=3)

        for itr2 in range(flags.batch_size):

            fig = plt.figure()
            pos = 240 + 1
            plt.subplot(pos)
            plt.imshow(valid_images[itr2].astype(np.uint8))
            plt.axis('off')
            plt.title('Original')

            pos = 240 + 2
            plt.subplot(pos)
            # plt.imshow(valid_annotations[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(valid_annotations[itr2].astype(
                np.uint8), cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
            plt.axis('off')
            plt.title('GT')

            pos = 240 + 3
            plt.subplot(pos)
            # plt.imshow(pred[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(pred[itr2].astype(np.uint8),
                       cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
            plt.axis('off')
            plt.title('Prediction')

            # Confusion matrix for this image prediction
            crossMat = EvalMetrics.calculate_confusion_matrix(
                valid_annotations[itr2].astype(
                    np.uint8), pred[itr2].astype(
                    np.uint8), num_classes)
            cross_mats.append(crossMat)

            np.savetxt(save_dir +
                       "Crossmatrix" +
                       str(itr1 *
                           flags.batch_size +
                           itr2) +
                       ".csv", crossMat, fmt='%4i', delimiter=',')

            # Save input, gt, pred, crf_pred, sum figures for this image

            """ Generate CRF """
            # 1. run CRF
            crfwithprobsoutput = denseCRF.crf_with_probs(
                valid_images[itr2].astype(np.uint8), predprob[itr2], num_classes)

            # 2. show result display
            crfwithprobspred = crfwithprobsoutput.astype(np.uint8)

            # -----------------------Save inp and masks----------------------
            Utils.save_image(valid_images[itr2].astype(np.uint8), save_dir,
                             name="inp_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(valid_annotations[itr2].astype(np.uint8), save_dir,
                             name="gt_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(pred[itr2].astype(np.uint8),
                             save_dir,
                             name="pred_" + str(itr1 * flags.batch_size + itr2))
            Utils.save_image(crfwithprobspred, save_dir,
                             name="crf_" + str(itr1 * flags.batch_size + itr2))

            # ----------------------Save visualized masks---------------------
            Utils.save_visualized_image(valid_annotations[itr2].astype(np.uint8), save_dir,
                                        image_name="gt_" + str(itr1 * flags.batch_size + itr2), n_classes=num_classes)
            Utils.save_visualized_image(pred[itr2].astype(np.uint8),
                                        save_dir,
                                        image_name="pred_" + str(itr1 * flags.batch_size + itr2), n_classes=num_classes)
            Utils.save_visualized_image(crfwithprobspred, save_dir, image_name="crf_" + str(
                itr1 * flags.batch_size + itr2), n_classes=num_classes)

            # --------------------------------------------------

            # Confusion matrix for this image prediction with crf
            prob_crf_crossMat = EvalMetrics.calculate_confusion_matrix(
                valid_annotations[itr2].astype(
                    np.uint8), crfwithprobsoutput.astype(
                    np.uint8), num_classes)
            crf_cross_mats.append(prob_crf_crossMat)

            np.savetxt(save_dir +
                       "prob_crf_Crossmatrix" +
                       str(itr1 *
                           flags.batch_size +
                           itr2) +
                       ".csv", prob_crf_crossMat, fmt='%4i', delimiter=',')

            pos = 240 + 4
            plt.subplot(pos)
            # plt.imshow(crfwithprobsoutput.astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
            plt.imshow(crfwithprobsoutput.astype(np.uint8),
                       cmap=ListedColormap(label_colors_10k), norm=clothnorm_10k)
            plt.axis('off')
            plt.title('Prediction + CRF')

            plt.savefig(save_dir + "resultSum_" +
                        str(itr1 * flags.batch_size + itr2))

            plt.close('all')
            print("Saved image: %d" % (itr1 * flags.batch_size + itr2))

    try:
        total_cm = np.sum(cross_mats, axis=0)
        np.savetxt(
            flags.logs_dir +
            "Crossmatrix.csv",
            total_cm,
            fmt='%4i',
            delimiter=',')

        # print("\n>>> Prediction results (TF functions):")
        # print("Pixel acc:", np.nanmean(tf_pixel_acc_list))
        # print("mean IoU:", np.nanmean(tf_miou_list))

        print("\n>>> Prediction results:")
        EvalMetrics.calculate_eval_metrics_from_confusion_matrix(total_cm, num_classes)

        # Prediction with CRF
        crf_total_cm = np.sum(crf_cross_mats, axis=0)
        np.savetxt(
            flags.logs_dir +
            "CRF_Crossmatrix.csv",
            crf_total_cm,
            fmt='%4i',
            delimiter=',')

        print("\n")
        print("\n>>> Prediction results (CRF):")
        EvalMetrics.calculate_eval_metrics_from_confusion_matrix(crf_total_cm, num_classes)

    except Exception as err:
        print(err)

    end = time.time()
    print("Testing time:", end - start, "seconds")

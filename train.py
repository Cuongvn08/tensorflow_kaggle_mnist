# -*- coding: utf-8 -*-

import shutil
import time
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

#import data as da
from data import Data
from model import Model
#import config as conf
from config import *
from datetime import datetime


################################################################################
## HELPFUL FUNCTIONS
################################################################################
## get batches
# @param data_len: data length (number of examples)
# @param batch_size: batch size (number of examples per training run)
# return: zip of batch starts and ends
def get_batches(data_len, batch_size):
    batch_starts = range(0, data_len, batch_size)
    batch_ends = [batch_start + batch_size for batch_start in batch_starts]
    return zip(batch_starts, batch_ends)

## get loss
# @param logit: logit from model [None, num_classes]
# @param label: class index [None]
def get_loss(logit, label):
    label = tf.to_int64(label)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                        logits=logit,
                                                        labels=label,
                                                        name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy)

    return loss

## get optimizer
# @param learning_rate:
# @param optimizer: optimizer method
def get_optimizer(learning_rate, optimizer):
    if optimizer == eOptimizer.Adam:
        return tf.train.AdamOptimizer(learning_rate = learning_rate,
                                      beta1 = 0.9,
                                      beta2 = 0.999,
                                      epsilon = 1e-10,
                                      use_locking = False,
                                      name = 'Adam')
    elif optimizer == eOptimizer.GD:
        return tf.train.GradientDescentOptimizer(learning_rate = learning_rate,
                                                 use_locking = False,
                                                 name = 'GradientDescent')
    elif optimizer == eOptimizer.RMS:
        return tf.train.RMSPropOptimizer(learning_rate = learning_rate,
                                         decay = 0.9,
                                         momentum = 0.0,
                                         epsilon = 1e-10,
                                         use_locking = False,
                                         centered = False,
                                         name = 'RMSProp')
    else:
        assert '[train] optimizer error'

## print and log
def print_log(logger, str):
    print(str)
    logger.write(str + '\n')
    logger.flush()

def get_confusion_matrix(pred, label):
    num_classes = 10
    confusion_matrix = np.zeros( (num_classes, num_classes), dtype=np.int)

    for (pred_idx, label_idx) in zip(pred, label):
        confusion_matrix[pred_idx, label_idx] += 1

    return confusion_matrix

def get_accuracy(confusion_matrix):
    sum = np.sum(confusion_matrix)
    true_pred = np.sum(confusion_matrix[i][i] for i in range(len(confusion_matrix[0])))
    return true_pred/sum

def get_precision(confusion_matrix):
    # do nothing
    pass

def get_recall(confusion_matrix):
    # do nothing
    pass

def get_F1score(confusion_matrix):
    # do nothing
    pass

################################################################################
## MAIN PROGRAM
################################################################################
# create log file
with tf.name_scope('logger'):
    log_path = 'result/log.txt'
    if os.path.exists(log_path):
        os.remove(log_path)
    
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    logger = open(log_path, 'w')
    print_log(logger, 'kaggle_mnist_competition')

# create session
with tf.name_scope('session'):
    sess = tf.Session()

# get config
with tf.name_scope('config'):
    cfig = get_config()

# load data
with tf.name_scope('data'):
    data = Data()

    data.read_train_data(cfig[eKey.train_path], cfig[eKey.train_ratio])

    train_data  = data.get_train_data()
    train_label = data.get_train_label()

    eval_data  = data.get_eval_data()
    eval_label = data.get_eval_label()

# create placeholders
with tf.name_scope('placeholder'):
    data = tf.placeholder(cfig[eKey.data_dtype],
                          shape=[None, 28, 28, 1], name='data')
    label = tf.placeholder(cfig[eKey.label_dtype], shape=[None],name='label')

    tf.add_to_collection('data', data)
    tf.add_to_collection('label', label)

    print_log(logger, '[train] shape of data placeholder {0}'.format(data.get_shape()))
    print_log(logger, '[train] shape of label placeholder {0}'.format(label.get_shape()))

# create model
with tf.name_scope('model'):
    model = Model()

# get train opt
with tf.name_scope('train'):
    train_logit = model.logit(data, True, cfig[eKey.dropout])
    train_cost = get_loss(train_logit, label)
    train_opt = get_optimizer(cfig[eKey.learning_rate], cfig[eKey.optimizer]).minimize(train_cost)

    train_pred = tf.argmax(tf.nn.softmax(train_logit), axis=1, name='train_pred')
    train_equal = tf.equal(train_pred, label)
    train_acc = tf.reduce_mean(tf.cast(train_equal, tf.float32))

    train_summary_list = []
    train_summary_list.append(tf.summary.scalar('train_cost', train_cost))
    train_summary_list.append(tf.summary.scalar('train_acc', train_acc))
    train_summary_merge = tf.summary.merge(train_summary_list)

# get eval opt
with tf.name_scope('eval'):
    tf.get_variable_scope().reuse_variables()
    eval_logit = model.logit(data, False)
    eval_cost = get_loss(eval_logit, label)

    eval_pred = tf.argmax(tf.nn.softmax(eval_logit), axis=1, name='pred')
    eval_equal = tf.equal(eval_pred, label)
    eval_acc = tf.reduce_mean(tf.cast(eval_equal, tf.float32))

    eval_summary_list = []
    eval_summary_list.append(tf.summary.scalar('eval_cost', eval_cost))
    eval_summary_list.append(tf.summary.scalar('eval_acc', eval_acc))
    eval_summary_merge = tf.summary.merge(eval_summary_list)

    tf.add_to_collection('pred', eval_pred)

# initialize variables
with tf.name_scope('initialize_variables'):
    sess.run(tf.global_variables_initializer())

# create summary
with tf.name_scope('summary'):
    if os.path.exists(cfig[eKey.log_dir]) is True:
        shutil.rmtree(cfig[eKey.log_dir])
    os.makedirs(cfig[eKey.log_dir])

    summary_writer = tf.summary.FileWriter(cfig[eKey.log_dir], sess.graph)

# create saver
with tf.name_scope('saver'):
    if os.path.exists(cfig[eKey.checkpoint_dir]) is True:
        shutil.rmtree(cfig[eKey.checkpoint_dir])
    os.makedirs(cfig[eKey.checkpoint_dir])

    saver = tf.train.Saver(max_to_keep = None)

# train
train_batches = get_batches(train_data.shape[0], cfig[eKey.batch_size])
for start, end in train_batches:
    print_log(logger, '[train] train_batches: start={0}, end={1}'.format(start, end))
    
logger.write('\n')    
with tf.name_scope('train'):
    with tf.device('/cpu:%d' % 0):
    #with tf.device('/gpu:%d' % 0):
        for step in range(cfig[eKey.num_epoch]):
            # train
            start_time = time.time()
            train_fetches = [train_logit, train_cost, train_opt, train_summary_merge]

            train_costs = []
            train_batches = get_batches(train_data.shape[0], cfig[eKey.batch_size])
            for start, end in train_batches:
                feed_dict = {}
                feed_dict[data] = train_data[start:end]
                feed_dict[label] = train_label[start:end]
                [_, tCost, _, tSummary] = sess.run(train_fetches, feed_dict)
                train_costs.append(tCost)

            # eval
            if step % cfig[eKey.eval_step] == 0:
                eval_fetches = [eval_logit, eval_cost, eval_pred, eval_equal, eval_acc, eval_summary_merge]

                feed_dict = {}
                feed_dict[data] = eval_data
                feed_dict[label] = eval_label
                [_, eCost, ePred, eEqual, eAcc, eSummary] = sess.run(eval_fetches, feed_dict)

                # log and print
                elapsed_time = time.time() - start_time
                date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                str = '[train] {0} step {1:04}: tCost = {2:0.5}; eCost = {3:0.5}; eAcc = {4:0.5}; time = {5:0.5}(s);'.\
                    format(date_time, step, np.mean(train_costs), eCost, eAcc, elapsed_time)
                print_log(logger,str)

                # get confusion matrix
                confusion_matrix = get_confusion_matrix(ePred, eval_label)
                if eAcc > 0.99:
                    print(confusion_matrix)

                # save summaries
                summary_writer.add_summary(tSummary, step)
                summary_writer.add_summary(eSummary, step)
                summary_writer.flush()

                # save checkpoint
                saver.save(sess, cfig[eKey.checkpoint_dir] + 'chk_step_%d.ckpt'%step)

                '''
                # show 3 random predictions
                idx = random.sample(range(eval_data.shape[0]), 3)

                plt.subplot(131)
                plt.imshow(np.squeeze(eval_data[idx[0]]), cmap='gray')
                plt.title('label = {0}; pred = {1}'.format(eval_label[idx[0]], ePred[idx[0]]))

                plt.subplot(132)
                plt.imshow(np.squeeze(eval_data[idx[1]]), cmap='gray')
                plt.title('label = {0}; pred = {1}'.format(eval_label[idx[1]], ePred[idx[1]]))

                plt.subplot(133)
                plt.imshow(np.squeeze(eval_data[idx[2]]), cmap='gray')
                plt.title('label = {0}; pred = {1}'.format(eval_label[idx[2]], ePred[idx[2]]))

                plt.pause(0.1)
                '''

    summary_writer.close()

plt.show()
logger.write('The end.')
logger.close()
print('The end.')

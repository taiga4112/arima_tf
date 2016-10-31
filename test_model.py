# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import mlp
import horse_data

#
LEARNING_LOG_DIR = "log"
TRAIN_DATA_DIR = "data"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
DATA_SIZE = 150
MAX_ORDER = 16
#

#
HIDDEN_LAYER_SIZE = 100
TRAINNING_SIZE = 2001
BATCH_SIZE = 20
#

## Set result model name ##
import sys
args = sys.argv
if len(args) < 2:
	print 'Error : Please select learned model.'
RESULT_MODEL = args[1]
##

# Variable
x = tf.placeholder("float", shape=(None, DATA_SIZE)) # 馬データを入れる仮のTensor
y_ = tf.placeholder("float", shape=(None, MAX_ORDER)) # 順位情報を入れる仮のTensor
w_h = tf.Variable(tf.random_normal([DATA_SIZE, HIDDEN_LAYER_SIZE], mean=0.0, stddev=0.05))
w_o = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE, MAX_ORDER], mean=0.0, stddev=0.05))
b_h = tf.Variable(tf.zeros([HIDDEN_LAYER_SIZE]))
b_o = tf.Variable(tf.zeros([MAX_ORDER]))

# model
y_hypo = mlp.model(x, w_h, b_h, w_o, b_o)

# modelの読み込み
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(init)
saver.restore(sess, RESULT_MODEL)

test_horse,test_label = horse_data.get_horse_and_label(TEST_FILE)
for i in range(len(test_horse)):
	pred = (y_hypo.eval(feed_dict={x: test_horse})[i])
	print np.argmax(pred)+1, np.argmax(test_label[i])+1
	print pred

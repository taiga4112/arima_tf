# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import tensorflow.python.platform


## Set result model name ##
import sys
args = sys.argv
RESULT_MODEL = "model.ckpt"
if len(args) < 2:
	print 'Warning : Result model will be saved by ''model.ckpt''.'
RESULT_MODEL = args[1]
##

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
TRAINNING_SIZE = 200001
BATCH_SIZE = 20
#

def get_horse_and_label(train_file):
	train_horse = []
	train_label = []
	f = open(train_file, 'r')
	for line in f:
		line = line.rstrip()
		l = line.split(",")
		# 馬情報の配列化と登録
		train_horse_file = TRAIN_DATA_DIR+"/"+l[0]
		train_horse_data = get_train_horse(train_horse_file)
		train_horse.append(train_horse_data)

		# 順位情報を1-of-K方式で用意する
		tmp = np.zeros(MAX_ORDER)
		tmp[int(l[1])-1] = 1
		train_label.append(tmp)

	f.close()
	train_horse = np.array(train_horse, dtype=np.float32)
	train_label = np.array(train_label, dtype=np.float32)
	return train_horse,train_label

def get_train_horse(train_horse_file):
	horse_data = []
	f = open(train_horse_file, 'r')
	for line in f:
		line = line.rstrip()
		# l = line.split(",")
		l = [float(x) for x in line.split(",")]
		horse_data.extend(l)
	f.close()
	return horse_data
	
def get_batch_data(numpy_array, start, size):
	return numpy_array[start:start+size]



def model(x, w_h, b_h, w_o, b_o):
	zh = tf.sigmoid(tf.matmul(x,w_h) + b_h)
	zo = tf.nn.softmax(tf.matmul(zh, w_o) + b_o)
	return zo

def loss(logits, labels):
	cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
	tf.scalar_summary("cross_entropy", cross_entropy) #  for TensorBoard
	return cross_entropy

def training(loss, learning_rate=0.001):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	return train_step

def accuracy(logits, labels):
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	tf.scalar_summary("accuracy", accuracy) #  for TensorBoard
	return accuracy

##############
####　main　####
##############
train_horse,train_label = get_horse_and_label(TRAIN_FILE)

# make learning log directory
if not os.path.exists(LEARNING_LOG_DIR):
	os.mkdir(LEARNING_LOG_DIR)# logフォルダを作成する

# Training!!
with tf.Graph().as_default():
	
	# Variable
	x = tf.placeholder("float", shape=(None, DATA_SIZE)) # 馬データを入れる仮のTensor
	y_ = tf.placeholder("float", shape=(None, MAX_ORDER)) # 順位情報を入れる仮のTensor
	w_h = tf.Variable(tf.random_normal([DATA_SIZE, HIDDEN_LAYER_SIZE], mean=0.0, stddev=0.05))
	w_o = tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE, MAX_ORDER], mean=0.0, stddev=0.05))
	b_h = tf.Variable(tf.zeros([HIDDEN_LAYER_SIZE]))
	b_o = tf.Variable(tf.zeros([MAX_ORDER]))

	# model
	y_hypo = model(x, w_h, b_h, w_o, b_o)

	# loss value
	loss_value = loss(y_hypo, y_)

	# 正則化は2乗ノルムを採用している
	L2_sqr = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)
	lambda_2 = 0.01
	loss = loss_value + lambda_2 * L2_sqr

	# trainning
	train_step = training(loss)

	# accuracy
	accuracy = accuracy(y_hypo, y_)


	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	# Tensorboard setting
	saver = tf.train.Saver()
	summary_op = tf.merge_all_summaries()
	summary_writer = tf.train.SummaryWriter(LEARNING_LOG_DIR+'/test_log', sess.graph_def)
	
	print('Trainning trainning!!')
	for step in range(TRAINNING_SIZE):

		for i in range(len(train_horse)/BATCH_SIZE):
			batch_xs = get_batch_data(train_horse,i,BATCH_SIZE)
			batch_ys = get_batch_data(train_label,i,BATCH_SIZE)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		
		# Consoleでの精度表示(2000stepごと)
		if step % 200 == 0:
			result = sess.run([summary_op, accuracy], feed_dict={x: train_horse, y_: train_label})
			summary_writer.add_summary(result[0], step)
			print('  step, accuracy = %6d: %6.3f' % (step, result[1]))

	# Test trained model
	test_horse,test_label = get_horse_and_label(TEST_FILE)
	test_accuracy = sess.run(accuracy, feed_dict={x: test_horse, y_: test_label})
	print('ACCURACY = %6.3f' % test_accuracy)

	# Save model
	save_path = saver.save(sess, RESULT_MODEL)

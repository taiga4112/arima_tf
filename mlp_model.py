# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

#
TRAIN_DATA_DIR = "data"
TRAIN_FILE = "train.txt"
DATA_SIZE = 150
MAX_ORDER = 17
#

#
HIDDEN_LAYER_SIZE = 100
TRAINNING_SIZE = 20001
BATCH_SIZE = 10
#

def get_train_horse_and_label(train_file):
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
		tmp[int(l[1])] = 1
		train_label.append(tmp)

	f.close()
	train_horse = np.asarray(train_horse)
	train_label = np.asarray(train_label)
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
	return cross_entropy

def training(loss, learning_rate=0.001):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	return train_step

def accuracy(logits, labels):
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	return accuracy


##############
####　main　####
##############
train_horse,train_label = get_train_horse_and_label(TRAIN_FILE)

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


# Training!!
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	print('Trainning trainning!!')
	for step in range(TRAINNING_SIZE):
		
		for i in range(len(train_horse)/BATCH_SIZE):
			batch_xs = get_batch_data(train_horse,i,BATCH_SIZE)
			batch_ys = get_batch_data(train_label,i,BATCH_SIZE)
			train_step.run({x: batch_xs, y_: batch_ys})
			# sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		
		if step % 2000 == 0:
			train_accuracy = accuracy.eval({x: batch_xs, y_: batch_ys})
			print('  step, accurary = %6d: %6.3f' % (step, train_accuracy))

	# Test trained model
	# print('accuracy = ', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))




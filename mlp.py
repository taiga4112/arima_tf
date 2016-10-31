# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

def model(x, w_h, b_h, w_o, b_o):
	zh = tf.sigmoid(tf.matmul(x,w_h) + b_h)
	zo = tf.nn.softmax(tf.matmul(zh, w_o) + b_o)
	return zo

def loss(logits, labels):
	# 以下は交差エントロピーを誤差関数として用いる場合
	# cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
	# tf.scalar_summary("cross_entropy", cross_entropy) #  for TensorBoard

	# 以下は2乗誤差関数を誤差関数として用いる場合
	_logits = tf.nn.softmax(logits)
	_labels = tf.nn.softmax(labels)
	loss = tf.nn.l2_loss(_logits - _labels)
	tf.scalar_summary("loss", loss) #  for TensorBoard
	return loss

def training(loss, learning_rate=0.001):
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	return train_step

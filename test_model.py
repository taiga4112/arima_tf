# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import mlp
import horse_data
import crawler

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


def get_final_order_array(pred_info):
	order_array = []
	order = 0
	while order < len(pred_info):
		max_number = 0
		max_pred = 0.0
		for i in range(len(pred_info)):
			if i not in order_array:
				pred = pred_info[i][order]
				if pred > max_pred:
					max_number = i
					max_pred = pred
		order_array.append(max_number)
		order = order + 1
	
	for i in range(len(order_array)):
		order_array[i] = order_array[i]+1

	return order_array


## Set result model name ##
import sys
args = sys.argv
if len(args) < 2:
	print 'Error : Please select learned model.'
RESULT_MODEL = args[1]
ARIMA_ID = args[2]
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
correct_result_array = crawler.get_all_race_data(ARIMA_ID)

# 尤度情報を記録
pred_info = []
for i in range(len(test_horse)):
	pred = (y_hypo.eval(feed_dict={x: test_horse})[i])
	pred_info.append(pred)

# 順位決定
order_array = get_final_order_array(pred_info)

# 正解データを整える
test_label_array = []
for i in range(len(order_array)):
	test_label_array.append(np.argmax(test_label[i])+1)

# わかりやすく予想結果と実際の順位を表示する
# for i in range(len(order_array)):
# 	print order_array[i], test_label_array[i]

# 今回の勉強会の点数を発表する
# 単勝5点買いのシミュレーション
cost = -500
for order in range(5):
	index = order_array.index(order+1)
	answer = test_label_array[index]
	horse_name = correct_result_array[answer-1][0]
	horse_number = correct_result_array[answer-1][2]
	odds = correct_result_array[answer-1][3]
	print int(horse_number), horse_name, order+1, answer, odds # [id, 馬名, 予想順位, 正解順位, 馬番, オッズ]
	if answer == 1:
		cost += 100*float(odds)
print cost


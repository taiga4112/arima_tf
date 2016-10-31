# -*- coding: utf-8 -*-
import os
import numpy as np

#
TRAIN_DATA_DIR = "data"
MAX_ORDER = 16
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
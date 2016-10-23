# -*- coding: utf-8 -*-
import sys
args = sys.argv

if len(args) < 2:
	print 'Error : Please add year of test data.'
	sys.exit()

test_year = args[1]

##
ALL_DATA_FILE = "data.txt"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
##

# 1. 訓練データとテストデータに分ける
train_data = []
test_data = []
f = open(ALL_DATA_FILE, 'r')
for line in f:
	line = line.rstrip()
	if line.find("_"+str(test_year)) > -1:
		test_data.append(line)
	else:
		train_data.append(line)
f.close()

# 2. 別々にファイルに保存
f = open(TRAIN_FILE, 'w')
for item in train_data:
	f.write("%s\n"%item)
f.close()

f = open(TEST_FILE, 'w')
for item in test_data:
	f.write("%s\n"%item)
f.close()
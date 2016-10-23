# -*- coding: utf-8 -*-

import urllib2
from bs4 import BeautifulSoup

###
DATA_DIR_NAME = "data"
PAGE_URL = "http://db.netkeiba.com/"
HORSE_RECORD = "horse/"
BLOOD_MARK = "horse/ped/"
RACE_MARK = "race/"
JOCKEY_MARK = "jockey/result/"
arima_race_record_url = PAGE_URL+"?pid=race_list&word=%CD%AD%C7%CF%B5%AD%C7%B0&track[]=1&list="
num_of_year_log = "30"
###

# 配列情報をファイルに出力
def write_csv1(file_path,array,mode="a"):
	fw = open(file_path,mode)
	for item in array:
		fw.write("%s\n"%item)
	fw.close()

# 配列情報をファイルに出力
def write_csv2(file_path,array2,mode="a"):
	fw = open(file_path,mode)
	for item in array2:
		fw.write("%s\n"%",".join(map(str,item)))
	fw.close()


# 指定したURLにある競走のidを取得する
def get_race_id_list(url):
	race_id_list = []
	html = urllib2.urlopen(url)
	soup = BeautifulSoup(html, "lxml")
	table = soup.find("table", class_="nk_tb_common race_table_01")
	rows = table.find_all('tr')
	for row in rows:
		col = row.find('td', class_="txt_l")
		if col != None:
			a = col.find('a')
			race_id_list.append(a['href'][6:-1])
	return race_id_list

# レース画面から[馬-順位]の情報を取得する
def create_race_data(race_id,  create_horse_data=True):

	year = race_id[:4]
	info_list = []

	url = PAGE_URL+RACE_MARK+race_id
	html = urllib2.urlopen(url)
	soup = BeautifulSoup(html, "lxml")
	table = soup.find("table", class_="race_table_01 nk_tb_common")
	rows = table.find_all('tr')

	number = 1
	for row in rows:
		col = row.find('td', class_="txt_l")
		if col != None:
			a = col.find('a')
			horse_id = a['href'][7:-1]
			horse_file_name = horse_id+"_"+year+".csv"
			info_list.append(horse_file_name+","+str(number))
			
			if create_horse_data:
				horse_record = extract_horse_record(get_horse_record(horse_id),year)
				write_csv2(DATA_DIR_NAME+"/"+horse_file_name, horse_record)

			number = number + 1
	
	write_csv1("data.txt",info_list)

# 馬の血統情報を抽出する
def get_horse_blood(horse_id):
	url = PAGE_URL+BLOOD_MARK+horse_id
	html = urllib2.urlopen(url)
	soup = BeautifulSoup(html, "lxml")
	table = soup.find("table", class_="blood_table detail")
	data = []
	items = table.find_all('td')
	for item in items:
		name = item.find('a').text.strip()
		data.append(name)
	return data

# 馬の戦績情報を抽出する
def get_horse_record(horse_id):
	print horse_id
	url = PAGE_URL+HORSE_RECORD+horse_id
	html = urllib2.urlopen(url)
	soup = BeautifulSoup(html, "lxml")
	table = soup.find("table", class_="db_h_race_results nk_tb_common")
	data_list = []
	table_body = table.find('tbody')
	rows = table_body.find_all('tr')
	for row in rows:
		data = [0.0]*30
		cols = row.find_all('td')
		data[0] = (int(cols[0].text.replace("/","")[4:]))# 日付
		data[1] = (get_place_id((cols[1].text)[1:3]))# レース会場
		data[2] = (get_weather_id(cols[2].text))# 天気
		data[3] = (get_race_grade_id(cols[4].text))# レースグレード
		data[4] = (get_float(cols[6].text))# 頭数
		data[5] = (get_float(cols[7].text))# 枠番
		data[6] = (get_float(cols[8].text))# 馬番
		data[7] = (get_float(cols[9].text))# オッズ
		data[8] = (get_float(cols[10].text))# 人気
		data[9] = (get_float(cols[11].text))# 着順

		# 騎手
		jockey_id = str(00)
		if cols[12].find('a') == None:# 登録されていないときの対応
			pass
		else:
			jockey_id = cols[12].find('a')['href'][8:-1]
			jockey_data = (get_jockey_info(jockey_id,str(cols[0].text.replace("/",""))[:4]))
			data[10] = jockey_data[0]
			data[11] = jockey_data[1]
			data[12] = jockey_data[2]
			data[13] = jockey_data[3]
			data[14] = jockey_data[4]
			data[15] = jockey_data[5]
			data[16] = jockey_data[6]
			data[17] = jockey_data[7]
			data[18] = jockey_data[8]
			data[19] = jockey_data[9]
			data[20] = jockey_data[10]
			data[21] = jockey_data[11]
			data[22] = jockey_data[12]
			data[23] = jockey_data[13]


		data[24] = (get_float(cols[13].text))# 斤量
		data[25] = (get_float((cols[14].text)[1:3]))# 距離
		data[26] = (get_time_second(cols[17].text))# タイム
		data[27] = (get_float(cols[18].text))# 着差
		data[28] = (get_float(cols[22].text))# 上り
		data[29] = (get_float((cols[23].text)[:2]))# 馬体重

		data_list.append(data)
	return data_list

# ジョッキーの情報を取得する
# その年度の[1着数,2着数,3着数]のリストにする
def get_jockey_info(jockey_id, year):
	j_info = []
	try:
		url = PAGE_URL+JOCKEY_MARK+jockey_id
		html = urllib2.urlopen(url)
		soup = BeautifulSoup(html, "lxml")
		table = soup.find("table", class_="nk_tb_common race_table_01")
		rows = table.find_all('tr')
		for row in rows:
			col = row.find('td', class_="txt_c")
			if col != None:
				if col.text == year:
					col2 = row.find('td', class_="txt_r")
					for i in range(0,14):
						col2 = col2.findNext('td')
						j_info.append(int(col2.find('a').text.strip()))
		
		if len(j_info)==0: # その年度の成績が存在しない（外人騎手など）場合
			for i in range(0,14):
				j_info.append(0)

	except:
		for i in range(0,14):
			j_info.append(0)
	return j_info

# 競馬場idを取得する
def get_place_id(place_name):
	place_id = 0
	try:
		if place_name == u"札幌":
			place_id = 1
		elif place_name == u"函館":
			place_id = 2
		elif place_name == u"福島":
			place_id = 3
		elif place_name == u"新潟":
			place_id = 4
		elif place_name == u"東京":
			place_id = 5
		elif place_name == u"中山":
			place_id = 6
		elif place_name == u"中京":
			place_id = 7
		elif place_name == u"阪神":
			place_id = 8
		elif place_name == u"京都":
			place_id = 9
		elif place_name == u"小倉":
			place_id = 10
	except:
		return 0
	return place_id

# レースのグレードidを取得する
def get_race_grade_id(race_name):
	grade_id = 0
	try:
		if race_name[-2] == u"1":
			grade_id = 1
		elif race_name[-2] == u"2":
			grade_id = 2
		elif race_name[-2] == u"3":
			grade_id = 3
	except:
		return 0
	return grade_id

# 天気idを取得する
def get_weather_id(weather_name):
	weather_id = 0
	try:
		if weather_name == u"晴":
			weather_id = 1
		elif weather_name == u"曇":
			weather_id = 2
		elif weather_name == u"雨":
			weather_id = 3
	except:
		return 0
	return weather_id

# float型に変換
def get_float(str_number):
	number = 0.0
	if str_number == "":
		return 0.0
	else:
		try:
			number = float(str_number)
		except:
			return 0.0
	return number

# レースタイムを秒単位に変換する
def get_time_second(race_time_str):
	s = 0.0
	if race_time_str == "":
		return 0.0
	else:
		try:
			s = float(race_time_str[0])*60 + float(race_time_str[2:])
		except:
			return 0.0
	return s

# 馬の戦績情報から、指定した部分のみを抽出する
def extract_horse_record(all_records_sorted_by_date,arima_year=2016,num_of_record=5):
	extracted_records = []
	extract_remain = num_of_record
	extract_flag = False
	for record in all_records_sorted_by_date:
		year = str(record[0])[:4]
		if extract_flag == False:
			if int(year)<=int(arima_year):
				extract_flag = True
		if extract_flag == True:
			if extract_remain == num_of_record:
				# その年の有馬記念
				pass
			elif extract_remain >= 0:
				extracted_records.append(record)
			
			extract_remain = extract_remain - 1
	return extracted_records



########################################
############### main ###################
########################################

import os
import datetime

if not os.path.exists(DATA_DIR_NAME):
	os.mkdir(DATA_DIR_NAME)# dataフォルダを作成する

race_id_list = get_race_id_list(arima_race_record_url+num_of_year_log)
number = 0
for race_id in race_id_list:
	print str(number), "/", len(race_id_list)-1, race_id, datetime.datetime.today()
	create_race_data(race_id)
	number = number + 1


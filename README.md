# arima_tf
TensorFlowを用いて、競馬予想をしてみるためのプログラムです.
ここでは、有馬記念を対象にデータの取得からニューラルネットの構築までを扱っています.

## Description
このプロジェクトには、以下の2つのソースコードが含まれています。
* crawler.py
   * [netkeiba.com](http://www.netkeiba.com/)から訓練データを取得するためのプログラム
* horse_data.py
   * 学習に必要な馬情報の構造を定義しているプログラム
* make_train_test.py
   * clawler.pyで取得したデータを訓練データとテストデータ(引数(西暦))に分割するプログラム
* mlp.py
   * TensorFlowで多層パーセプトロンを定義しているプログラム
* create_model.py
   * 訓練データを元に多層パーセプトロンを構築するプログラム
* test_model.py
   * create_model.pyによって構築した多層パーセプトロンをテストデータによって評価するプログラム

## Requirement
- pip (8.1.2)
- tensorflow (0.10.0)
- beautifulsoup4 (4.5.1)
- lxml (3.6.4)
- funcsigs (1.0.2)
- mock (2.0.0)
- python (2.7.10)
- numpy (1.11.1)
- glibc (0.6.1)
- pbr (1.10.0)
- protobuf (3.0.0b2)
- six (1.10.0)

## Usage
#### Mac or Linux (Ubuntu or CentOS7)
1. Fork it ([http://github.com/taiga4112/arima_tf/fork](http://github.com/taiga4112/arima_tf/fork))

2. Set developing environment (For Unix. Please check the official [page](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#virtualenv-installation))
	```bash
	$ cd 'yourworkspace'
	$ git clone git@github.com:youraccount/arima_tf.git
	$ virtualenv arima_tf
	$ source arima_tf/bin/activate
	$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
	$ pip install beautifulsoup4, lxml
	```

3. Collect data
	```bash
	$ cd arima_tf
	$ python crawler.py
	```

3. Divide data into training data and test data
	```bash
	$ python make_train_test.py 'TEST_DATA_YEAR(ex. 2015)'
	```

4. Develop multi-layer perceptron(MLP)
	```bash
	$ python create_model.py 'OUTPUT_MODEL_NAME(ex. model2015.ckpt)'
	```

5. Running TensorBoard if you want to check.
	```bash
	$ tensorboard --logdir log/test_log
	```

6. Check your learning result in [TensorBoard](http://localhost:6006).

7. Evaluating MLP by using test data
	```bash
	$ python test_model.py 'MODEL_NAME(ex. model2015.ckpt)' 'race_id(from netkeiba.com) ex.201506050810'
	```

## Contribution
1. Fork it (http://github.com/taiga4112/arima_tf/fork)
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create new Pull Request

## Author

[taiga4112](https://github.com/taiga4112)
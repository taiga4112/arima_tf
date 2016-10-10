# arima_tf
TensorFlowを用いて、競馬予想をしてみるためのプログラムです（個人的なテストです）.
ここでは、有馬記念を対象にデータの取得からニューラルネットの構築までを扱っています.

## Description
このプロジェクトには、以下の2つのソースコードが含まれています。
* crawler.py
   * [netkeiba.com](http://www.netkeiba.com/)から訓練データを取得するためのプログラム
* mlp_model.py
   * TensorFlowで多層パーセプトロンを構築するプログラム

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

2. Set developing environment
	```bash
	$ cd 'yourworkspace'
	$ git clone git@github.com:youraccount/arima_tf.git
	$ virtualenv arima_tf
	$ source arima_tf/bin/activate
	$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
	$ pip install beautifulsoup4, lxml
	```

3. Collect training data
	```bash
	$ cd arima_tf
	$ python crawler.py
	```

4. Develop multi-layer perceptron(MLP)
	```bash
	$ python mlp_model.py
	```


## Contribution
1. Fork it ( http://github.com/taiga4112/arima_tf/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create new Pull Request

## Author

[taiga4112](https://github.com/taiga4112)
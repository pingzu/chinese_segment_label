缺少相关文件需要依次运行 process_data.py, training.py, application.py 文件
使用的 Python 版本为 Python3, 神经网络框架为 Keras==2.1.2 
词性标注训练出的模型很大，180多M，文件里没法放，需要重新训练
data/segment/ 目录下 input.txt 为输入，output.txt 为分词输出
data/label 同理

python process_data.py segment			# 加工分词数据
python process_data.py label			# 加工标注数据
python process_data.py segment label	# 加工分词与标注数据
python training.py segment				# 训练分词模型
python training.py label				# 训练标注模型
python application.py label				# 测试标注
python application.py segment			# 测试分词
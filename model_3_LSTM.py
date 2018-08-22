#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 
import pickle as pkl 
import sys

seed = 123
tf.set_random_seed(seed)
np.random.seed(seed)

features = []
y = []

test_data = []
test_y = []

test_l = [1]
train_l = [0]
save_path = "for_scp/"

months = []
months_test = []
for i in train_l:
	for month_num in range(4):
		with open(save_path+"features_"+str(i)+"_"+str(month_num),'rb') as f:
			feature = pkl.load(f)
			months.append(feature)
		if month_num == 2:
			with open(save_path+"labels_"+"0"+"_"+str(month_num),'rb') as f:
				y_r = pkl.load(f)
				y = y_r

	for month_num in range(4):
		with open(save_path+"features_"+str(i)+"_"+str(month_num),'rb') as f:
			feature = pkl.load(f)
			months_test.append(feature)
		if month_num == 2:
			with open(save_path+"labels_"+"1"+"_"+str(month_num),'rb') as f:
				y_r = pkl.load(f)
				test_y = y_r
for i in range(len(months[0])):
	features.append([months[0][i],months[1][i],months[2][i],months[3][i]])
for i in range(len(months_test[0])):
	test_data.append([months_test[0][i],months_test[1][i],months_test[2][i],months_test[3][i]])
n_test = 0
for t_y in test_y:
	if t_y[0] == 1:
		n_test += 1

x = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)
month_num = 3

learning_rate = 0.001
layers = [features[0].shape[1],100,100,2]

b = [tf.Variable(tf.zeros(l,dtype=tf.float32)) for l in layers[1:]]
init_range = [np.sqrt(6.0/(w_s_1+w_s_2)) for w_s_1,w_s_2 in zip(layers[:-1],layers[1:])]
w = [tf.Variable(tf.random_uniform((w_s_1,w_s_2),minval=-init_range[i],maxval=init_range[i],dtype=tf.float32))
			for w_s_1,w_s_2,i in zip(layers[:-1],layers[1:],range(len(init_range)))]
w_for_memory = [tf.Variable(tf.random_uniform((m_s,m_s),minval=-1,maxval=1,dtype=tf.float32))
			for m_s in layers[1:-1]]
#2层，3序列
input_w = []
output_w = []
memory_w = []
forget_w = []
states = []


for i in range(len(layers)-2):
	input_w_row = []
	output_w_row = []
	memory_w_row = []
	forget_w_row= []
	#这样做只存每层当前的states
	states.append(np.zeros(layers[i+1]))
	for j in range(month_num):
		nput_w_row.append(tf.Variable(tf.zeros(layers[i+1])))
		#input_w_row.append(tf.Variable(tf.random_uniform(layers[i+1],minval=-1,maxval=-1,dtype=tf.float32)))
		output_w_row.append(tf.Variable(tf.random_uniform(layers[i+1],minval=-1,maxval=-1,dtype=tf.float32)))
		memory_w_row.append(tf.Variable(tf.random_uniform(layers[i+1],minval=-1,maxval=-1,dtype=tf.float32)))
		forget_w_row.append(tf.Variable(tf.random_uniform(layers[i+1],minval=-1,maxval=-1,dtype=tf.float32)))
	input_w.append(input_w_row)
	output_w.append(output_w_row)
	memory_w.append(memory_w_row)
	forget_w.append(forget_w_row)

for j in range(month_num):
	zs = []
	zs_plus = []
	activates = [x[j]]
	for i in range(len(layers)-2):
		zs.append(tf.reduce_sum(tf.multiply(activates[i],w[i].T),reduction_indices=1))
		memory_gate = tf.tanh(tf.reduce_sum(tf.multiply(zs[-1],memory_w[i][j]), reduction_indices=1))
		input_gate = tf.sigmoid(tf.reduce_sum(tf.multiply(zs[-1],input_w[i][j]), reduction_indices=1))
		forget_gate = tf.sigmoid(tf.reduce_sum(tf.multiply(zs[-1],forget_w[i][j]), reduction_indices=1))
		output_gate = tf.sigmoid(tf.reduce_sum(tf.multiply(zs[-1],output_w[i][j]), reduction_indices=1))
		states[i] = states[i]*forget_gate+memory_gate*input_gate
		add = tf.reduce_sum(tf.multiply(tf.tanh(states[i])*output_gate, w_for_memory[i]),reduction_indices=1)
		zs_plus.append(zs[-1]+add+b[i])
		activates.append(tf.nn.relu(zs_plus[-1]))
	predict_0 = tf.reduce_sum(tf.multiply(activates[-1],w[-1].T),reduction_indices=1)
	predict = tf.nn.softmax(predict_0+b[-1])

#---------------------

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=labels))/mini_batch_size
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
epochs = 20


for j in range(len(features)):
	states = []
	for i in range(layers-2):
		states.append(np.zeros(layers[i+1]))
	sess.run(train_step,feed_dict={x:features[j],labels:y[j]})

	predict_right_cout = 0
	loss_predict_right_cout = 0
	for i in range(len(test_y)): 
		predict_right_tf = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels,1)),"float"))
		predict_right_tf_2 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),2*tf.argmax(labels,1)),"float"))
		predict_right = sess.run(predict_right_tf,feed_dict={x:test_data[i],labels:test_y[i]})
		predict_right_2 = sess.run(predict_right_tf_2,feed_dict={x:test_data[i],labels:test_y[i]})
		if predict_right_2 == 1:
			loss_predict_right_cout += 1
		if predict_right == 1:
			predict_right += 1
	sys.stdout.write("{1}'s train data:{2} / {3}, how much we predict right: {4} / {5}".format(j, predict_right_2, n_test, predict_right, len(test_y)))
	sys.stdout.write("\r")
	sys.stdout.flush()
















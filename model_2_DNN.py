#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 
import pickle as pkl 
import csv
import sys

seed = 123
tf.set_random_seed(seed)
np.random.seed(seed)

features = np.array([])
y = np.array([])

test_data = np.array([])
test_y = np.array([])
n_test = 0


test_l = [0,4,8,10]
#test_l = [0]
train_l = [1,2,3,5,6,7,9]
#train_l = [0]
save_path = "for_scp/"
for i in train_l:
	with open(save_path+"features_"+str(i),'rb') as f:
		feature = pkl.load(f)
		if i == train_l[0]:
			features = feature
		else:
			features = np.concatenate((features,feature),axis=0)
	with open(save_path+"labels_"+str(i),'rb') as f:
		y_r = pkl.load(f)
		if i == train_l[0]:
			y = y_r
		else:
			y = np.concatenate((y,y_r),axis=0)
save_path = "for_scp/"
for i in test_l:
	with open("for_scp/"+"features_"+str(i),'rb') as f:
		feature = pkl.load(f)
		if i == test_l[0]:
			test_data = feature
		else:
			test_data = np.concatenate((test_data,feature),axis=0)
	with open(save_path+"labels_"+str(i),'rb') as f:
		y_r = pkl.load(f)
		if i == test_l[0]:
			test_y = y_r
		else:
			test_y = np.concatenate((test_y,y_r),axis=0)

n_test = 0
for y in y_r:
	if y == [1,0]:
		n_test += 1

x = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

mini_batch_size = 1000
mini_batches = [features[k:k+mini_batch_size] for k in range(0,len(features),mini_batch_size)]
mini_batches_y = [y[k:k+mini_batch_size] for k in range(0,len(features),mini_batch_size)]

learning_rate = 0.001
hidden_num = [features.shape[1],100,10,2]

b_shape = []
w_shape = []
init_range = []

w = [] 
b = []
activates = []
zs = [x]

for i in range(len(hidden_num)-1):
	b_shape.append(hidden_num[i+1])
	w_shape.append((hidden_num[i],hidden_num[i+1]))
	init_range.append(np.sqrt(6.0/(w_shape[i][0])+w_shape[i][1]))

	b.append(tf.Variable(tf.zeros(b_shape[i],dtype=tf.float32)))
	w.append(tf.Variable(tf.random_uniform(w_shape[i],minval=-init_range[i],maxval=init_range[i],dtype=tf.float32)))
	zs.append(tf.matmul(zs[i],w[i]))
	if i != len(hidden_num)-2:
		activates.append(tf.nn.relu(zs[-1]+b[i]))
predict = tf.nn.softmax(zs[-1]+b[-1])

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=labels))/mini_batch_size
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
epochs = 30

for i in range(epochs):
	for j in range(len(mini_batches)):
		sess.run(train_step,feed_dict={x:mini_batches[j],labels:mini_batches_y[j]})
		
		predict_right_tf = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels,1)),"float"))
		predict_right_tf_2 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),2*tf.argmax(labels,1)),"float"))
		predict_right = sess.run(predict_right_tf,feed_dict={x:test_data,labels:test_y})
		predict_right_2 = sess.run(predict_right_tf_2,feed_dict={x:test_data,labels:test_y})
		sys.stdout.write("Epochs {0}, {1}'s mini_batch:{2} / {3}, how much we predict right: {4} / {5}".format(i, j, predict_right_2, n_test, predict_right, len(test_y)))
		sys.stdout.write("\r")
		sys.stdout.flush()


	predict_right_tf = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(labels,1)),"float"))
	predict_right_tf_2 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),2*tf.argmax(labels,1)),"float"))
	predict_right = sess.run(predict_right_tf,feed_dict={x:test_data,labels:test_y})
	predict_right_2 = sess.run(predict_right_tf_2,feed_dict={x:test_data,labels:test_y})
	print("\nEpochs {0}:{1} / {2}. how much we predict right: {3} / {4}".format(i, predict_right_2, n_test, predict_right, len(test_y)))
















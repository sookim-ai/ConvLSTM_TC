from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from train import *
from testing import *
from rnn import *
import numpy as np
import skimage.measure
#State from ground truth, initial position:given, output location feed to input as mask
#(6, 60101, 256, 513)
h=256; w=513;
#1: Log files
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")

#2: Graph
#Training Parameters
validation_step=10;
learning_rate =0.002
X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels]) #shape=(24, ?, 256, 513, 6)
Y = tf.placeholder("float", [FLAGS.batch_size, None, h,w,1]) #shape=(24, ?, 256, 513, 1)
timesteps = tf.shape(X)[1]
h=tf.shape(X)[2] #h:256
w=tf.shape(X)[3] #w:513

prediction, last_state = ConvLSTM(X) #shape=(24, ?, 256, 513, 1)
loss_op=tf.losses.mean_pairwise_squared_error(Y,prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#3 :Training
with tf.Session() as sess:
    # Initialize all variables
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_X,train_Y,test_X,test_Y,val_X,val_Y=read_input("/Users/kim79/convLSTM/Model_in_aims-gpu/data/10/")
    for ii in range(1000):
        train(sess,loss_op,train_op,X,Y,train_X,train_Y,val_X,val_Y,prediction, last_state,fout_log)
        name=str(rep)+str(ii)+"_size10_"
        test(name,sess,loss_op,train_op,X,Y,test_X,test_Y,prediction,last_state,fout_log)
fout_log.close();

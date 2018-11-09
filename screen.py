import os
import numpy as np
from PIL import Image

scr_l = []
for f in os.listdir('./screen_set'):
	if f.endswith('jpg'):
		scr_l.append(os.path.join('./screen_set',f))
print len(scr_l)
images = np.array(Image.open(scr_l[0]).resize((128,128)).convert('L'))
images = images[np.newaxis,:,:]	
for i in scr_l[1:]:
	ima = np.array(Image.open(i).resize((128,128)).convert('L'))
	ima = ima[np.newaxis,:,:]
	images = np.concatenate((images,ima))
images = images[:,:,:,np.newaxis]
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
images = images.astype(np.float32)
images = np.multiply(images, 1.0 / 255.0)

import tensorflow as tf
def weight_variable(shape,n):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial,name=n)

def bias_variable(shape,n):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial,name=n)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') 

def max_pool_4x4(x):
	return tf.nn.max_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	
x = tf.placeholder(tf.float32,[16384])
x_image = tf.reshape(x,[1,128,128,1])
W_conv1 = weight_variable([5,5,1,10],'conv_1/weights/W_conv_1')
b_conv1 = bias_variable([10],'conv_1/biases/b_conv_1')
W_conv2 = weight_variable([5,5,10,20],'conv_2/weights/W_conv_2')
b_conv2 = bias_variable([20],'conv_2/biases/b_conv_2')
W_conv3 = weight_variable([5,5,20,40],'conv_3/weights/W_conv_3')
b_conv3 = bias_variable([40],'conv_3/biases/b_conv_3')
W_fc1 = weight_variable([8*8*40,100],'fc_1/weights/W_fc1')
b_fc1 = bias_variable([100],'fc_1/biases/b_fc1')
W_fc2 = weight_variable([100,2],'fc_2/weights/W_fc2')
b_fc2 = bias_variable([2],'fc_2/biases/b_fc_2')

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,'./models/w_and_b/data.ckpt')

W_conv1_T = sess.run(W_conv1)
b_conv1_T = sess.run(b_conv1)
W_conv2_T = sess.run(W_conv2)
b_conv2_T = sess.run(b_conv2)
W_conv3_T = sess.run(W_conv3)
b_conv3_T = sess.run(b_conv3)
W_fc1_T = sess.run(W_fc1)
b_fc1_T = sess.run(b_fc1)
W_fc2_T = sess.run(W_fc2)
b_fc2_T = sess.run(b_fc2)



h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1_T)+b_conv1_T)
h_pool1 = max_pool_4x4(h_conv1)	
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2_T)+b_conv2_T)
h_pool2 = max_pool_2x2(h_conv2)	
h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3_T)+b_conv3_T)
h_pool3 = max_pool_2x2(h_conv3)
h_pool3_flat = tf.reshape(h_pool3,[-1,8*8*40])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1_T)+b_fc1_T)
y = tf.nn.softmax(tf.matmul(h_fc1,W_fc2_T)+b_fc2_T)

for i in range(len(scr_l)):
	pb = sess.run(y,feed_dict={x:images[i]})
	r = np.array(pb)
	if r[0,1]>r[0,0]:
		print scr_l[i], r

sess.close()


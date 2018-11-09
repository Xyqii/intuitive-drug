import predatasets
import tensorflow as tf

moles = predatasets.get_data()

with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32,[None,16384],name='graphs')
	y_ = tf.placeholder(tf.float32,[None,2],name='type')

with tf.name_scope('in_tensors'):	
	x_image = tf.reshape(x,[-1,128,128,1])

def weight_variable(shape,n):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial,name=n)

def bias_variable(shape,n):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial,name=n)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def max_pool_4x4(x):
	return tf.nn.max_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

with tf.name_scope('conv_1'):
	with tf.name_scope('weights'):
		W_conv1 = weight_variable([5,5,1,10],'W_conv_1')
	with tf.name_scope('biases'):
		b_conv1 = bias_variable([10],'b_conv_1')
	with tf.name_scope('convolution_and_non_linear'):
		h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
	with tf.name_scope('pool'):
		h_pool1 = max_pool_4x4(h_conv1)		              
with tf.name_scope('conv_2'):
	with tf.name_scope('weights'):
		W_conv2 = weight_variable([5,5,10,20],'W_conv_2')
	with tf.name_scope('biases'):
		b_conv2 = bias_variable([20],'b_conv_2')
	with tf.name_scope('convolution_and_non_linear'):
		h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) 
	with tf.name_scope('pool'):
		h_pool2 = max_pool_2x2(h_conv2)
with tf.name_scope('conv_3'):
	with tf.name_scope('weights'):
		W_conv3 = weight_variable([5,5,20,40],'W_conv_3')
	with tf.name_scope('biases'):
		b_conv3 = bias_variable([40],'b_conv_3')
	with tf.name_scope('convolution_and_non_linear'):
		h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
	with tf.name_scope('pool'):
		h_pool3 = max_pool_2x2(h_conv3)
with tf.name_scope('flatten'):
	h_pool3_flat = tf.reshape(h_pool3,[-1,8*8*40])
with tf.name_scope('fc_1'):
	with tf.name_scope('weights'):
		W_fc1 = weight_variable([8*8*40,100],'W_fc1')
	with tf.name_scope('biases'):
		b_fc1 = bias_variable([100],'b_fc1')
	with tf.name_scope('matmul_and_non_linear'):
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)
with tf.name_scope('fc_2'):
	with tf.name_scope('weights'):
		W_fc2 = weight_variable([100,2],'W_fc2')
	with tf.name_scope('biases'):
		b_fc2 = bias_variable([2],'b_fc_2')
with tf.name_scope('outputs'):
	y = tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)
with tf.name_scope('cross_entorpy'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
	tf.summary.scalar('cross_entorpy',cross_entropy)
with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

with tf.name_scope('evaluate'):
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("graph/",sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


for i in range(1001):
	batch_xs, batch_ys = moles.train.next_batch(400)
	sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})	
	if i% 20 == 0:
		result = sess.run(merged,feed_dict={x:batch_xs,y_:batch_ys})	
		writer.add_summary(result,i)
		print sess.run(accuracy,feed_dict={x:moles.test.images,y_:moles.test.labels})
			

save_path = saver.save(sess,'w_and_b/data.ckpt')

sess.close()




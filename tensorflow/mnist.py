import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

try:
	xrange
except NameError:
	xrange = range
  
mnist = input_data.read_data_sets("data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
   
sess = tf.Session()
sess.run(tf.initialize_all_variables())
   
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for step in xrange(500):
	batch_xs, batch_ys = mnist.train.next_batch(1000)
	sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
	if step % 20 == 0:
		# print(step, sess.run(W), sess.run(b))
		print("after step ", step)
		print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print("after train")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

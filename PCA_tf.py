import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import tensorflow as tf
import os

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

#task 01
Num = '3' # the digit to be processed
P = 2 # the feature dimension, only 2 is supported at the moment
path = '.' + os.sep + 'optdigits-orig.tra' + os.sep + 'optdigits-orig.tra'
input_file = open(path)
sample = []
flag = 1
for line in islice(input_file,21,None,1):
    line = line.strip('\n')
    line = line.strip(' ')
    if line == Num:
        data_list = np.array(list(map(int, sample)))
        data_list = data_list[ :, np.newaxis]
        sample = []
        if flag:
            data = data_list;
            flag = 0
        else:
            data = np.hstack((data, data_list))
    elif len(line) == 1:
        sample = []
    else:
        sample = sample + list(line)

#task 02
meandata = np.mean(data, axis =1)
meandata = meandata[ :, np.newaxis]
meanpic = meandata.reshape(32,32)
data_n = (data - meandata).transpose();
row = np.size(data_n,0)
col = np.size(data_n,1)
 
# Model parameters
x1 = weight_variable([1, col])
x2_1 = weight_variable([1, col-1])
x2_2 = tf.div(-tf.matmul(x1[:,0:(col-1)],tf.transpose(x2_1)), x1[0,col-1])
x2 = tf.concat([x2_1, x2_2], axis=1)
x = tf.concat([x1, x2], axis=0)

W1 = weight_variable([row, 1])
W2_1 = weight_variable([row -1, 1])
W2_2 = tf.div(-tf.matmul(tf.transpose(W1[0:(row -1),:]),W2_1), W1[row-1,0])
W2 = tf.concat([W2_1, W2_2], axis=0)
W = tf.concat([W1, W2], axis=1)

# Model input and output
linear_model = tf.matmul(W,  x) 
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y) ) # sum of the squares

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(0.04)
optimizer =tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

# training data# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
loss_new = 0.0
for i in range(100000):
  if i%100 == 0:
      loss_old = loss_new
      loss_new = sess.run(loss, {y:data_n})
      if (abs(loss_old - loss_new) < 0.1):
        break
      print("index: %s loss: %s"%(i, loss_new))
  sess.run(train, {y:data_n})

W_r = sess.run(W)
x_r = sess.run(x)
xx =  tf.matmul(x, tf.transpose(x))
xx_r = sess.run(xx)

x_r = x_r/np.sqrt([[xx_r[0,0]],[xx_r[1,1]]])
W_r = W_r*np.sqrt([xx_r[0,0], xx_r[1,1]])

# rearrange the principal components and its coordinates
W_variance = np.std(W_r, axis = 0)
order = P - 1 -  np.argsort(W_variance)
W_r = W_r[:, order]
x_r = x_r[order, :]

plt.figure(1)
coordinate1 =  np.tile([-5, -2.5, 0.0, 2.5, 5], 5)
temp = np.arange(5, -6, -2.5)
temp = temp[ :, np.newaxis]
coordinate2 =  np.matmul(temp,np.ones((1,5)))
coordinate2 = coordinate2.reshape(25,)
plt.scatter(coordinate1, coordinate2, color='', marker='o', edgecolors='r')

plt.plot(W_r[:,0], W_r[:,1], 'yo',markersize=2) 
plt.ylabel('Second Principal Component')
plt.xlabel('First Principal Component')
plt.grid(color='k', linestyle='--', linewidth=1)
#plt.savefig('1_tf.png', dpi=300)

plt.figure(2)
for index in range(1,26):
    plt.plot([1,2,3])
    components = np.matmul(np.array([coordinate1[index-1], coordinate2[index-1]]), x_r)
    components = components[ :, np.newaxis]
    pic_pca = np.add(meandata, components)
    #pic_pca = meandata
    pic_pca = 1 - pic_pca
    pic_pca =  (pic_pca >= 0.5) * 1 ## binary image
    pic_pca = pic_pca.reshape(32,32)
    plt.subplot(5,5,index)
    plt.axis('off') 
    plt.imshow(pic_pca, cmap="gray")
#plt.savefig('2_tf.png', dpi=300)
plt.show()



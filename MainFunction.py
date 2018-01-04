import tensorflow as tf
import numpy.random as rng
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image


def weight_variable(shape,std):
  initial = tf.truncated_normal(shape,stddev=std)
  var= tf.Variable(initial)
  return var

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')

def create_pairs(x, digit_indices):
    """ Positive and negative pair creation.
        Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(40)]) -1
    for d in range(40):
        for i in range(n):
            indices_same=random.sample(range(i+1, 10),int((10-i-1)))
            for indice in indices_same:
                z1, z2 = digit_indices[d][i], digit_indices[d][indice]
                pairs += [[x[z1], x[z2]]]
                labels+=[1]

            indices_catogory = random.sample(range(0, 40), 5)
            for dn in indices_catogory:
                if dn!= d:
                    indices_diff = random.sample(range(i + 1, 10), int((10-i-1)/2))
                    for indice in indices_diff:
                        z1, z2 = digit_indices[d][i], digit_indices[dn][indice]
                        pairs += [[x[z1], x[z2]]]
                        labels += [0]
    return np.array(pairs), np.array(labels)


#parameters of input image
train_image=[]
train_label=[]
valid_image=[]
valid_label=[]
test_image=[]
test_label=[]


# Read ATT face dataset
dir='att_faces'

#image buffer to store all images
image=[]
#label buffer to store all labels with same order as image
label=[]
data_label=[]
data_image=[]

for dirname, dirnames, filenames in os.walk('att_faces'):
    # print path to all filenames
    for filename in filenames:
        print(dirname)
        number=0
        if dirname[-3]=='s':
            number=int(dirname[-2:])
        else:
            number=int(dirname[-1])
        label.append(number)
        data_label=np.asarray(label)
        print(os.path.join(dirname, filename))
        X = Image.open(os.path.join(dirname, filename))
        image.append(np.array(X).transpose())
        data_image = np.asarray(image)

data_image = data_image.astype('float32')
data_image /= 255



# create training+test positive and negative pairs
digit_indices = [np.where(data_label == i)[0] for i in range(1,41)]
image_pairs, label_pairs = create_pairs(data_image, digit_indices)
label_pairs=label_pairs.reshape([label_pairs.size,1])

#random shuffle the data set
randomize = np.arange(len(label_pairs))
np.random.shuffle(randomize)
image_pairs = image_pairs[randomize,:,:,:]
label_pairs = label_pairs[randomize]


#split into training/ validation and test set
split=[8,1,1]
split_point1=int(image_pairs.shape[0]* split[0]/10)
split_point2=int(image_pairs.shape[0]* (split[0]+split[1])/10)
split_point3=int(image_pairs.shape[0])

train_image=image_pairs[0:split_point1,:,:,:]
valid_image=image_pairs[split_point1+1 : split_point2,:,:,:]

test_image=image_pairs[split_point2+1 :split_point3,:,:,:]

train_label=label_pairs[0:split_point1]
valid_label=label_pairs[split_point1+1 : split_point2]
print('Valid set: the number of same pairs are '+ str(np.sum(valid_label)))
print('Valid set: the number of pairs are '+str(valid_label.shape[0]))
test_label=label_pairs[split_point2+1:split_point3]

#release the memory
image_pairs=[]
label_pairs=[]
digit_indices=[]
data_image=[]
data_label=[]




nb_epoch = 1
batchsize=16
beta=0.5







x1 = tf.placeholder(tf.float32, shape=[None, 92,112])

x2 = tf.placeholder(tf.float32, shape=[None, 92,112])
x_image = tf.reshape(x1, [-1, 92,112, 1])
x_ref = tf.reshape(x2, [-1,92,112, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])




#first conv layer
W_conv1 = weight_variable([5, 5, 1, 256],0.2)
b_conv1 = bias_variable([256])

#second conv layer
W_conv2 = weight_variable([3, 3, 256, 128],np.sqrt(2)/(3*np.sqrt(256)))
b_conv2 = bias_variable([128])


#Third conv layer
W_conv3 = weight_variable([3, 3, 128, 128],np.sqrt(2)/(3*np.sqrt(128)))
b_conv3 = bias_variable([128])

#densely connected layer
W_fc1 = weight_variable([12 * 14 * 128, 2048],np.sqrt(2)/(13*np.sqrt(128)))
b_fc1 = bias_variable([2048])
keep_prob = tf.placeholder(tf.float32)
W_fc2 = weight_variable([2048, 1],0.01)
b_fc2 = bias_variable([1])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


h_pool3_flat = tf.reshape(h_pool3, [-1, 12*14*128])

x1_full = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)



h_conv1 = tf.nn.relu(conv2d(x_ref, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool3_flat = tf.reshape(h_pool3, [-1, 12*14*128])
x2_full = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)



y_conv = tf.matmul(np.abs(x1_full- x2_full), W_fc2) + b_fc2





regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2)

loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv))+beta*regularizers/batchsize
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

y_conv=tf.round(tf.nn.sigmoid(y_conv))

accuracy=tf.reduce_mean(tf.cast(tf.equal(y_,y_conv),tf.float32))


Max_accuracy=0
loss_over_training=[]
accuracy_over_training=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(nb_epoch):
        start=0
        for i in range(5000):

            batch_x1=train_image[start:start+batchsize,0,:,:]
            batch_x2 = train_image[start:start + batchsize,1,:,:]
            batch_y=train_label[start:start + batchsize,]


            start += batchsize
            if start>=train_label.size:
                start= 0

            train_step.run(feed_dict={x1: batch_x1, x2: batch_x2, y_: batch_y, keep_prob: 1})

            if i % 200 == 0:
                loss_value = loss.eval(feed_dict={
                    x1: valid_image[:, 0], x2: valid_image[:, 1], y_: valid_label, keep_prob: 1.0})
                loss_over_training=np.append(loss_over_training,loss_value)
                test_accuracy=accuracy.eval(feed_dict={
                    x1: valid_image[:, 0], x2: valid_image[:, 1], y_: valid_label, keep_prob: 1.0})
                accuracy_over_training = np.append(accuracy_over_training, test_accuracy)

                training_accuracy = accuracy.eval(feed_dict={
                    x1: batch_x1, x2: batch_x2, y_: batch_y, keep_prob: 1.0})
                print('step %d, loss %g' % (i, loss_value))
                print('step %d, training_accuracy is %g' % (i, training_accuracy))
                print('step %d, test_accuracy is %g' % (i, test_accuracy))

                Max_accuracy=np.maximum(test_accuracy,Max_accuracy)
        print('Maximum Accuracy is %g' % (Max_accuracy))
    train_accuracy = accuracy.eval(feed_dict={
        x1: test_image[:, 0], x2: test_image[:, 1], y_: test_label, keep_prob: 1.0})
    print('final test_accuracy is %g' % (train_accuracy))

    test_accuracy = accuracy.eval(feed_dict={
        x1: test_image[:, 0], x2: test_image[:, 1], y_: test_label, keep_prob: 1.0})
    print('final test_accuracy is %g' % (test_accuracy))

plt.figure(1)
plt.plot(loss_over_training)
plt.title('The validation set loss： ')
plt.show()

plt.figure(2)
plt.plot(accuracy_over_training)
plt.title('The validation accuracy: ')
plt.show()
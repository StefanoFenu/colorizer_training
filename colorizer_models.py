from sklearn import metrics, cross_validation
from os import listdir
from skimage.io import imread
import skflow
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb
import numpy as np
import pickle

def load_data():

    root_data_dir = 'color_data/all/'
    output_dir = 'output/'

    color_images = None
    gray_images  = None
    
    print 'Loading data:'

    percent_done = -1
    subset = listdir(root_data_dir)[:1000]

    for i, img_file in enumerate(subset):

        percent = 100*i / len(subset)

        if percent_done != percent:
            print '\t\t' + str(percent) + '%'
            percent_done = percent

        image = resize(imread(root_data_dir + img_file ), (28,28))
        gray_image = rgb2gray(image).flatten()
        flat_image = image.flatten()
        
        if len(flat_image) > len(gray_image):

            if color_images == None or gray_images == None:
                gray_images = np.array([gray_image])
                color_images = np.array([flat_image])
            else:
                gray_images = np.vstack((gray_images,gray_image))
                color_images = np.vstack((color_images, flat_image))

    X_train = gray_images[:len(gray_images)/2]
    y_train = color_images[:len(color_images)/2]
    X_test  = gray_images[len(gray_images)/2:]
    y_test  = color_images[len(color_images)/2:]


    return X_train, y_train, X_test, y_test

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def basic_conv_net():
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    Y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    w_conv1 = weight_variable([5,5,1,32])
    b_conv1 = vias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2_flat, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    for i in tange(20000):
        batch = next_batch(50)
        if not i%100:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob:1.0
                })
            print('step %d, train accuracy %g'%(i, train_accuracy))
        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    print('test accuracy %g'%accuracy.eval(feed_dict={x:test_images, y_:test_labels, keep_prob:1.0}))

def loss_function_averager():

    return


if __name__ == '__main__':


    root_data_dir = 'color_data/all/'
    output_dir = 'output/'

    color_images = None
    gray_images  = None
    
    print 'Loading data:'

    percent_done = -1
    subset = listdir(root_data_dir)[:30000]

    for i, img_file in enumerate(subset):

        percent = 100*i / len(subset)

        if percent_done != percent:
            print '\t\t' + str(percent) + '%'
            percent_done = percent

        image = resize(imread(root_data_dir + img_file ), (28,28))
        gray_image = rgb2gray(image).flatten()
        flat_image = image.flatten()
        
        if len(flat_image) > len(gray_image):

            if color_images == None or gray_images == None:
                gray_images = np.array([gray_image])
                color_images = np.array([flat_image])
            else:
                gray_images = np.vstack((gray_images,gray_image))
                color_images = np.vstack((color_images, flat_image))

    X_train = gray_images[:len(gray_images)/2]
    y_train = color_images[:len(color_images)/2]
    X_test  = gray_images[len(gray_images)/2:]
    y_test  = color_images[len(color_images)/2:]

    print 'Generating test split'


    print 'Starting to train'
    classifier = skflow.TensorFlowDNNRegressor(hidden_units=[1024, 2048, 2048, 3072], batch_size=128, verbose=1, optimizer='SGD', steps=20000)


    classifier.fit(X_train, y_train)
   
    classifier.save('models/skflow_09/')


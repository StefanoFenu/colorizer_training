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

        image = resize(imread(root_data_dir + img_file ), (32,32))
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

if __name__ == '__main__':


    root_data_dir = 'color_data/all/'
    output_dir = 'output/'

    color_images = None
    gray_images  = None
    
    print 'Loading data:'

    percent_done = -1
    subset = listdir(root_data_dir)[:10000]

    for i, img_file in enumerate(subset):

        percent = 100*i / len(subset)

        if percent_done != percent:
            print '\t\t' + str(percent) + '%'
            percent_done = percent

        image = resize(imread(root_data_dir + img_file ), (32,32))
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
    classifier = skflow.TensorFlowDNNRegressor(hidden_units=[1024,2048, 3072],
            batch_size=128, verbose=1, optimizer='SGD', steps=20000)


    classifier.fit(X_train, y_train)
   
    classifier.save('models/skflow_06/')


import skflow
from skimage.io import imread, imsave
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


if __name__ == '__main__':

    root_classifier_dir = 'models/skflow_06/'
    classifier = skflow.TensorFlowDNNRegressor.restore(root_classifier_dir)
    test_image_file = 'test.jpg'
    test_image = rgb2gray(resize((imread(test_image_file)), (32,32))).flatten()

    foo = np.array([test_image, test_image, test_image, test_image] )
    output = classifier.predict(np.array([test_image]))
    output = np.reshape(output, (32,32,3))
    imsave('test_output.jpg', output)


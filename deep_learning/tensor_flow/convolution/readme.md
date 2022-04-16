https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/
https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/?completed=/loading-custom-data-deep-learning-python-tensorflow-keras/

Creates a neural networks using tensorflow and Keras.
The code will
- Read images from a directory.  The names of the subdirectories are classification categories, and the files are images.
- Pickles the numpy-ed arrays.  In subsequent runs, if the files exist then they are used.
- Creates a neural network using keras/tensorflow.
- Fits the dataset

Arguments
- [1] - A relative or absolute path to train the neutral network.  This directory must contain subdirectories for each class
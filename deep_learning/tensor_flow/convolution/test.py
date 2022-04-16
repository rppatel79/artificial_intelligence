import pathlib
import random

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D

TMP_DIR = "tmp/"
IMG_SIZE = 400


def resize_image(img_array):
    return cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


def show_image(img_array):
    plt.imshow(img_array, cmap='gray')
    plt.show()


def create_training_data(data_dir: str):
    image_and_classnum = []

    class_num = 0
    for category in os.listdir(data_dir):
        if pathlib.Path(os.path.join(data_dir,category)).is_file():
            break
        print("The category ["+category+"] has class_num ["+str(class_num)+"]")

        class_num = class_num+1
        path = os.path.join(data_dir, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img),
                                       cv2.IMREAD_GRAYSCALE)
                image_and_classnum.append([resize_image(img_array), class_num])

            except Exception as e:
                print("Issue while processing ["+img+"]"+str(e))

    random.shuffle(image_and_classnum)

    X = []
    y = []
    for image, classnum in image_and_classnum:
        X.append(image)
        y.append(classnum)
    return X, y


def file_name(data_type: str, is_x: bool):
    if (is_x):
        return TMP_DIR + data_type + ".X.pickle"
    else:
        return TMP_DIR + data_type + ".y.pickle"


def write_files(data_type: str, X, y):
    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    pickle_X_file_name = file_name(data_type, True)

    pickle_out = open(pickle_X_file_name, "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_y_file_name = file_name(data_type, False)
    pickle_out = open(pickle_y_file_name, "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def load_files(data_type: str):
    pickle_in = open(file_name(data_type, True), "rb")
    X = pickle.load(pickle_in)

    pickle_in = open(file_name(data_type, False), "rb")
    y = pickle.load(pickle_in)

    return X, y


if __name__ == '__main__':
    args = sys.argv[1:]
    data_dir = args[0]

    base_path = os.path.basename(data_dir)
    if pathlib.Path(file_name(base_path,True)).is_file() and pathlib.Path(file_name(base_path,False)).is_file():
        print("Loading cached set from [", file_name(base_path,True), ",", file_name(base_path,False),"]")
        X, y = load_files(base_path)
    else:
        print("Loading data set [", data_dir, "]")
        X, y = create_training_data(data_dir)
        # show_image(X[0])

        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        y = np.array(y)

        write_files(base_path, X, y)

    X = X / 255.0

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X, y, batch_size=4, validation_split=0.1, epochs=3)

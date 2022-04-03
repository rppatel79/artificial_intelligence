import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# names of the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.imshow(test_images[0],cmap=plt.cm.binary)
# plt.show()
#print(class_names[test_labels[0]])

# Scales the values to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# flatten the data

# build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Test the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("test_loss", test_loss, ", test_acc", test_acc)

predictions = model.predict(np.array(test_images))
#predictions = model.predict(np.array([test_images[9]]))

plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Action: "+class_names[test_labels[i]])
    plt.title("Prediction: "+class_names[np.argmax(predictions[i])])
    plt.show()
# Import Libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# Split data into testing and training data - 'using keras'
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# To define labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
				'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boots']

# To modify the images '/250' because they are stored in numpy array
train_images = train_images/255.0
test_images = test_images/255.0

print(train_images[7])

"""
# show image using matplotlib
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show
"""

# Create a model to connect to layers
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),	
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax") 
	])

# Set up parameters for model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

"""
# Train model
model.fit(train_images, train_labels, epochs=7)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc:", test_acc)
"""

# Using a model to make a prediction / and also show image
prediction = model.predict(test_images)

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual: " + class_names[test_labels[i]])
	plt.title("Prediction" + class_names[np.argmax(prediction[i])])
	plt.show()
import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

#print(train_data[0])

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Reverse dictionary
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# To make everything the same size
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index[<pad>], padding=post, maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index[<pad>], padding=post, maxlen=250)

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review(test_data[0]))

# Create our model

model = keras.Sequential()
model.add(keras.layerts.Embedding(1000, 16))
model.add(keray.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()
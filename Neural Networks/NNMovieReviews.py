import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)

print(train_data[0])

#grabs the tuples
word_index = data.get_word_index()

#k for key, v for value
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#swap all values and keys
reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])


def decode_review(text):
  return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review(test_data[0]))
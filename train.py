from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import tensorflow as tf

import pandas as pd
import numpy as np

from pythainlp import word_tokenize

import pickle


# Read caption file and generate token dataframe
df = pd.read_csv("caption.txt", names=["text"])
df["token"] = df["text"].apply(
    lambda text: word_tokenize(text, keep_whitespace=True))
df["processed_token"] = df["token"].apply(lambda tokens: tokens + ["end"])
df["processed_sentence"] = df["processed_token"].apply(
    lambda tokens: " ".join(tokens))


# Print head of dataframe
# print(df.head())


# Tokenization
tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(df["processed_sentence"])
total_words = len(tokenizer.word_index) + 1


# Print tokenized word (Word in dictionary)
# print(tokenizer.word_index)


# Convert sentences to sequence of tokens
input_sequences = []
for line in df["processed_sentence"]:
    token_list = tokenizer.texts_to_sequences([line])[0]

    # Cropping sequence for all possibility
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


# Make padded sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))
predictors = input_sequences[:, :-1]
label = input_sequences[:, -1]
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

print(
    f"Max words / sentence: {max_sequence_len}, Total words in dictionary: {total_words}")


# Create model
model = Sequential()
model.add(Embedding(total_words, output_dim=128,
          input_length=max_sequence_len - 1))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


# Training
model.fit(predictors, label, epochs=500, verbose=1)


# Save the model
model.save("model.h5")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

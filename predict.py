from keras.preprocessing.sequence import pad_sequences
from pythainlp import word_tokenize
import numpy as np
import pickle
import tensorflow as tf

# Define sequnce length
max_sequence_len = 28

# Input starting text
input_text = input()

# Load model
model = tf.keras.models.load_model('model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Make word dictionary
word_dictionary = {token: word for word, token in tokenizer.word_index.items()}

# Make seed text
output_text = " ".join(word_tokenize(input_text))

# Loop to generate sequence
for i in range(max_sequence_len):
    token_list = tokenizer.texts_to_sequences([output_text])[0]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted = np.random.choice(
        np.arange(0, predicted.shape[1]), p=predicted[0])

    output_word = word_dictionary[predicted]

    output_text += " " + output_word

    if output_word == "end":
        output_text = "".join(output_text.split(" ")[:-1])


# Print output text
print(output_text)

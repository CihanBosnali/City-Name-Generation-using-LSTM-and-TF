from __future__ import absolute_import, division, print_function

import os
from six import moves

import ssl

import tflearn
from tflearn.data_utils import *

# Get data
data_path = "US_cities.txt"

if not os.path.isfile(data_path):
    context = ssl._create_unverified_context()
    # Get US names dataset
    moves.urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt", data_path, context=context)

# Maximum length of generated names
maxlen = 20

# Translate text file to vectors
X, Y, char_idx = textfile_to_semi_redundant_sequences(data_path, seq_maxlen=maxlen, redun_step=3)

# Create LSTM model
model = tflearn.input_data(shape=[None, maxlen, len(char_idx)])

model = tflearn.lstm(model, 512, return_seq=True)

model = tflearn.dropout(model, 0.5)

model = tflearn.lstm(model, 512)

model = tflearn.dropout(model, 0.5)

model = tflearn.fully_connected(model, len(char_idx), activation="softmax")

# Generate city names
model = tflearn.SequenceGenerator(model,
                                  dictionary=char_idx,
                                  seq_maxlen=maxlen,
                                  clip_gradients=5.0,
                                  checkpoint_path="model_us_cities")

# training
for i in range(40):
    seed = random_sequence_from_textfile(data_path, maxlen)
    model.fit(X, Y, validation_set=0.2, batch_size=128, n_epoch=1, run_id='us_cities')

    print("Testing 0.5:", model.generate(30, temperature=0.5,seq_seed=seed))
    print("Testing 1.0:", model.generate(30, temperature=1.0,seq_seed=seed))
    print("Testing 1.2:", model.generate(30, temperature=1.2,seq_seed=seed))
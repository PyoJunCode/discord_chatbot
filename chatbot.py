#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import re
import urllib.request
import time
import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import os
import json
import requests



def take_input(input):

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('tokenizer')
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    VOCAB_SIZE = tokenizer.vocab_size + 2

    max_len = 100
    vocab_size = 2**13


    parse = encoding(input,START_TOKEN,END_TOKEN)
    data = json.dumps({"signature_name": "serving_default", "instances":parse.tolist()})


    headers = {'content-type': 'application/json'}
    json_response = requests.post('http://localhost:8501/v1/models/chatbot:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']


    return (np.argmax(predictions))


def encoding(sentence,START_TOKEN,END_TOKEN):

    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    sentence = tf.keras.preprocessing.sequence.pad_sequences(
      sentence, maxlen=max_len, padding='post')


    return sentence



def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence

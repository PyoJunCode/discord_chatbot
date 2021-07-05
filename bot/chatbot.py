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
import random
import discord
import boto3
from botocore.exceptions import ClientError

from discord.ext import commands


class Chatbot(commands.Cog):

    def __init__(self,bot):
        self.bot = bot
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('./cogs/tokenizer')
        self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        self.max_len = 100
        self.vocab_size = 2**13
        self.apikey = 'V7IEX00EYGO7'


    @commands.command(name="반응")
    async def take_input(self,ctx, arg):

        VOCAB_SIZE = self.tokenizer.vocab_size + 2

        parse = self.encoding(arg)
        data = json.dumps({"signature_name": "serving_default", "instances":parse.tolist()})

        #Change localhost to AWS serving server
        headers = {'content-type': 'application/json'}
        json_response = requests.post('http://localhost:8501/v1/models/chatbot:predict', data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']

        category = np.argmax(predictions)

        rnd = random.choice(range(0,2))

        searchterm = {0:'daily',1:'comfort',2:'love', 3:'angry', 4:'anxiety', 5:'sad'}

        lmt = 8
        r = requests.get(
        "https://g.tenor.com/v1/search?q=%s&key=%s&limit=%s" % (searchterm[category], self.apikey, lmt))

        if r.status_code == 200:
            # load the GIFs using the urls for the smaller GIF sizes
            top_8gifs = json.loads(r.content)
            results = top_8gifs['results']

            idx =  random.randrange(0,8)
            await ctx.send(results[idx]['url'])
        else:
            top_8gifs = None


    def encoding(self,sentence):

        sentence = self.preprocess_sentence(sentence)

        sentence = tf.expand_dims(
          self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)

        sentence = tf.keras.preprocessing.sequence.pad_sequences(
          sentence, maxlen=self.max_len, padding='post')

        return sentence


    def preprocess_sentence(self,sentence):
      sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
      sentence = sentence.strip()
      return sentence


    @commands.command(name="업로드")
    async def upload(self,ctx):

        file = 'newdadta.pickle'
        name = 'newdata.pickle'

        s3 = boto3.client(
        's3',  # instance name
        aws_access_key_id="YOUR_ID",         # ID
        aws_secret_access_key="YOUR_KEY")    # Secret_key

        s3.upload_file(file,"BUCKET_NAME_YOU_WANT",name)


def setup(bot):
    bot.add_cog(chatbot(bot))


# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import keras
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.layers.embeddings import Embedding
import pickle

with open('./tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


# In[6]:


data0=pd.read_csv('./test.csv')
pd.set_option('display.max_colwidth',-1)
X1 = tokenizer.texts_to_sequences(data0['Scene'].values)

max_length_of_text=200

X1 = pad_sequences(X1, maxlen=max_length_of_text)


# In[7]:


from keras.models import load_model
model = load_model('./model.h5')

batch_size=32


dic = {
    0:'POSTIVE',
    1:'NEGATIVE',
    2:'MIXED',
    3:'NEUTRAL'
}

preds = model.predict(X1, batch_size=batch_size)
preds = np.argmax(preds, axis=1)

preds = [ dic.get( preds[index] )   for index in range(len(preds)) ]


# In[8]:


new_df=pd.DataFrame({"Index":[i+1 for i in range(len(preds))],"Sentiment":preds})
new_df.to_csv("./solution.csv",index=False)


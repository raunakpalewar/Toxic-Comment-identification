 
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import nltk
# nltk.download('all')

 
df = pd.read_csv("./jigsaw-toxic-comment-classification-challenge/train.csv/train1.csv")

 
df.head()

 
from tensorflow.keras.layers import TextVectorization
import string
import html
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# nltk.download('all')
stop_words = set(stopwords.words('english'))

def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


def to_lowercase(text):
    return text.lower()



def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    return re.sub(r'\d+', '', text)
def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
    """
    :param words:
    :type words:
    :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    or
    from spacy.lang.en.stop_words import STOP_WORDS
    :type stop_words:
    :return:
    :rtype:
    """
    return [word for word in words if word not in stop_words]


def stem_words(words):
    """Stem words in text"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_words(words):
    """Lemmatize words in text"""

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]
def lemmatize_verbs(words):
    """Lemmatize verbs in text"""

    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

def text2words(text):
    return word_tokenize(text)

def clean_text( text):
    text = remove_special_chars(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    words = text2words(text)
    words = remove_stopwords(words, stop_words)
    #words = stem_words(words)# Either stem ovocar lemmatize
    words = lemmatize_words(words)
    words = lemmatize_verbs(words)

    return ''.join(words)

 
X = df['comment_text']
y = df[df.columns[2:]].values

 
X= X.apply(lambda x: clean_text(x))

 
MAX_FEATURES = 200000 # number of words in the vocab

 
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')

 
vectorizer.adapt(X.values)

 
vectorized_text = vectorizer(X.values)

 
#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks

 
train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

  

 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

 
model = Sequential()
# Create the embedding layer 
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(6, activation='sigmoid'))


model.compile(loss='BinaryCrossentropy', optimizer='Adam', metrics=['accuracy'])


model.summary()

 
history = model.fit(train, epochs=2, validation_data=val, )

import tensorflow as tf
import gradio as gr
# import os

def set_result(group,tox,high):

    f=open('./static/result.txt','w')
    # print('open')
    f.write('<p>1. Your Submitted Comment comes under <b>'+group+'</b>.</p>')
    if((tox+high)/2 >80):
        f.write('<p>2.The above sentence seems to be highly toxic and it is recommended to avoid using it and also report it immediately to the particular consent.!!!</p>')
    elif((tox+high)/2 >65):
        f.write('<p>2.The preceding statement above appears to be mildly toxic and is recommended to avoid using it as possible.!!!</p>')  
    else:
        f.write('<p>2.The above preceding assertion seems to have ignorable toxicity and thus not mandatory to avoid it.!!!</p>')          
    f.write('<p>3. Percentage of Toxicity : <b>'+str(tox)+'</b> </p><p>4.Percentage of '+group+' : <b>'+str(high)+'</b> </p>')
    f.close()

    return 0

def set_good_result():

    f=open('./static/result.txt','w')
    print('open')
    f.write('<p>COMMENT IS <b>NON TOXIC</b>.</p>')
    f.close()

    return 0


def resultAPI(comment):
    results = model.predict(vectorizer([clean_text(comment)]))
    text = ''
    text2 = ''
    tox = 0
    high=0
    group = ''
    flag=0
    for idx, col in enumerate(df.columns[2:]):
        text2 += '{}: {} '.format(col, results[0][idx]>0.5)
        if (results[0][idx]>0.5):
            text2 += '   - {}% \n'.format(round(results[0][idx]*100 , 2))
            if (round(results[0][idx]*100 , 2)>high and flag==1):
                high=round(results[0][idx]*100 , 2)
                group= col
            if (flag==0):
                tox=round(results[0][idx]*100 , 2)
        else:
            text2 += '\n'
        flag=1
    if(tox>0):
        set_result(group, tox, high)
    else:
        set_good_result()

    return text2



  

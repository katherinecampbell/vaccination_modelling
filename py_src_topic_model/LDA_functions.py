'''
Created on 10 Oct 2018

@author: kec52
'''

import os
import sys
sys.stdout = open('/home/kec5204/nohup.out', 'w')
import logging
from time import time
import pickle
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import gensim 
from gensim import corpora, models, similarities, utils
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import MmCorpus, Dictionary
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pyLDAvis.gensim
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.text import Text

def get_stops():
    with open ('/home/kec5204/latin_stop.txt', 'r') as latin:
        latin_stop = [word.strip(",.") for line in latin for word in line.lower().split()]
        latin_stop.append('ab')

    with open ('/home/kec5204/german_stop.txt', 'r') as german:
        german_stop = [word.strip(",.") for line in german for word in line.lower().split()]
        german_stop.append('aber')
    
    with open ('/home/kec5204/french_stop.txt', 'r') as french:
        french_stop = [word.strip(",.") for line in french for word in line.lower().split()]
        french_stop.append('alors')
    
    with open ('/home/kec5204/spanish_stop.txt', 'r') as spanish:
        spanish_stop = [word.strip(",.") for line in spanish for word in line.lower().split()]
        spanish_stop.append('un')

        english_stop = list(STOPWORDS)

        extra_stop = ['tli', 'thk', 'http', 'org', 'ofthe', 'tha', 'tho', 'ther', 'there', 'der', 'dat', 'ain', 'tis', 'thee', 'thou', 'thy', 'waa']
        stop_words = english_stop + extra_stop + spanish_stop + german_stop + latin_stop + french_stop
        return stop_words
 #lemmatize for verbs
# def get_lemma(word):
#     lemma = wn.(word)
#     if lemma is None:
#         lemma = word
#     return lemma

#lemmatize for nouns
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def tokenize(text):
    stop_words = get_stops()
    return [token for token in simple_preprocess(text) if token not in stop_words]

#clean texts
def clean_txt(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2]
#     tokens = [get_lemma(token) for token in tokens]
    tokens = [get_lemma2(token) for token in tokens]
    return tokens

def iter_documents(top_directory):
    """Iterate over all documents, yielding a document (=list of utf8 tokens) at a time."""
    count = 0
    for root, dirs, files in os.walk(top_directory):
        for file in filter(lambda file: file.endswith('.txt'), files):
            document = io.open(os.path.join(root, file), encoding='utf=8', errors='ignore').read() # read the entire document, as one big string
            x = clean_txt(document) # or whatever tokenization suits
            yield x



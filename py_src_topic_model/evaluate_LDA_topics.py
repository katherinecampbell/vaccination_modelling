#!/usr/bin/python
'''
Created on 10 Oct 2018

@author: kec52
'''
from corpus_class import MyCorpus
from LDA_functions import *
import os
import sys
#sys.stdout = open('/home/kec5204/nohup.out', 'w')
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


 
logging.basicConfig(format= '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#stop word from txt files to lists

#returns corpus, serialized corpus, and dict. args: top_dir, corpus_name
#saves serialized corpus mm and dictionary as .dict
#both args in ''

def get_corpus_dict(top_dir, corpus_name):
    start = time()
    print('starting')
    corpus = MyCorpus(top_dir)
    print('corpus made')
    #save corpus
    pickle.dump(corpus, open(corpus_name + '.pkl', 'wb'))
    print('corpus saved')
    #save dictionary
    dictionary = corpus.dictionary
    print('dictionary saved')
    dictionary.save(corpus_name +'_dictionary.dict')
    new_corpus = [vector for vector in iter(corpus)]
    corpora.MmCorpus.serialize(corpus_name+'_serialized.mm', new_corpus)
    print ('used: {:.2f}s'.format(time()-start))
    # Building reverse index.
    for (token, uid) in dictionary.token2id.items():
        dictionary.id2token[uid] = token
    return corpus, new_corpus, dictionary


def lda_model(corpus,dictionary, num_passes, num_topics, chunksize):
    print('making model')
    start = time()
    LDA = gensim.models.ldamodel.LdaModel(corpus, id2word = dictionary, passes = num_passes, num_topics = num_topics, chunksize=chunksize)
    #print time
    print('used: {:.2f}s'.format(time()-start))
    return LDA


#topic coherence - human interpretability of topic model using cv coherence
# arguments: dictionary = Gensim dictionary, corpus =  Gensim corpus, limit = max num topics
# Returns: lm_list = List of LDA topic models and c_v  = Coherence values corresponding to LDA model with respective num_topics

def evaluate_graph(dictionary, corpus, passes, chunksize, limit):
    print('begin')
    c_v = []
    lm_list = []
    sys.stdout.write('beginning eval...')
    print('beginning eval...')
    for num_topics in range(30, limit):
        sys.stdout.write("num topics tested is %d" % num_topics)
        print("num topics tested is", num_topics)
        lm = lda_model(corpus, dictionary, passes, num_topics, chunksize)
        print('lm made')
        lm_list.append(lm)
        print(lm_list)
        cm = CoherenceModel(model=lm, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        c_v.append(cm.get_coherence())
        print('appended')
        
    # Show graph
    sys.stdout.write('making graph...')
    print('making graph')
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    fig1 = plt.gcf()
    plt.show()
    plt.draw
    fig1.savefig('coherence_plot.png')
    sys.stdout.write('fig saved')
    return lm_list, c_v


corpus = pickle.load(open('/home/kec5204/notebooks/full_corpus.pkl','rb'))
corpus_dict = Dictionary.load('/home/kec5204/notebooks/full_corpus_dictionary.dict')
lm, cv = evaluate_graph(corpus_dict, corpus, 2, 1000, 70)
print(cv)

print('DONE!!!!!')
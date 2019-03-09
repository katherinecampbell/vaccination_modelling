'''
Created on 10 Oct 2018

@author: kec52
'''

import gensim 
from LDA_functions import iter_documents
from gensim import corpora, models, similarities, utils
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import MmCorpus, Dictionary
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import os

class MyCorpus(object):
    
    def __init__(self, top_dir):
        self.top_dir = top_dir
        self.dictionary = gensim.corpora.Dictionary(iter_documents(top_dir))
        self.dictionary.filter_extremes(no_below=20, keep_n=30000) # check API docs for pruning params
    
    def __len__(self):
        count = 0
        for root, dirs, files in os.walk(self.top_dir):
            for file in filter(lambda file: file.endswith('.txt'), files):
                count += 1
        self._data_len = int(count)
        return self._data_len
    
    def __iter__(self):
        for tokens in iter_documents(self.top_dir):
            yield self.dictionary.doc2bow(tokens)

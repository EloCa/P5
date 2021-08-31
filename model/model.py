# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:42:36 2021

@author: elodi
"""

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
import joblib
from skmultilearn.model_selection import iterative_train_test_split
import re


def check_tag(word, list_tags):
    for t in list_tags:
        if word == t:
            simple_t = re.match('(\.?[a-z]+)', t).group()
            return (True, simple_t)
    return (False, '')


def simplify_tags_sentence(s, list_tags):
    l = []
    for w in s.split(' '):
        b, x = check_tag(w, list_tags)
        if b:
            l.append(x)
    return l

def remove_stopwords_sentence(sentence):
    # print(sentence)
    return pd.Series([word for word in sentence[0].split() if word not in stop])


def remove_stopwords_df(df):
    return df.apply(remove_stopwords_sentence, axis=1)


def stem_sentence(s):
    stemmer = EnglishStemmer()
    return pd.Series([stemmer.stem(w) for w in s if not pd.isna(w)])




# load the data associated with programming languages tags
df_prog = pd.read_csv('D:/OP/P5/df_prog.csv')

# load the programming languages tags
df_tags = pd.read_csv('D:/OP/P5/tags_programming_languages.csv')
list_tags = df_tags.Tags.to_list()

stop = stopwords.words('english')
punctuation = string.punctuation

# body and title pipeline
body_pipeline = Pipeline(steps=[
    ('remove html tags', FunctionTransformer(pd.DataFrame.replace,
                                             kw_args={'to_replace': '<.*?>', 'value': '', 'regex': True})),
    ('lower', FunctionTransformer(lambda x: x.squeeze(axis=1).str.lower().to_frame())),
    ('remove punctuation', FunctionTransformer(lambda x: x.squeeze(
        axis=1).str.replace('[{}]'.format(punctuation), '').to_frame())),
    ('remove stopwords', FunctionTransformer(remove_stopwords_df, validate=False)),
    ('stemming', FunctionTransformer(pd.DataFrame.apply, kw_args={
     'func': stem_sentence, 'axis': 1}, validate=False)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
    #('vectorizer', CountVectorizer(lowercase=False,analyzer='word', preprocessor=None, tokenizer=lambda i:i ))
])

# tags
tags_pipeline = Pipeline(steps=[
    ('remove <>', FunctionTransformer(lambda x: x.str.extractall(r'<(.*?)>').groupby(level=0).agg({0: ' '.join}))),
    ('lower', FunctionTransformer(lambda x: x.squeeze(axis=1).str.lower().to_frame())),
    ('select_tags', FunctionTransformer(lambda x: x.squeeze(axis=1).apply(simplify_tags_sentence, list_tags=list_tags).to_frame())),
    #('split', FunctionTransformer(lambda x: x.squeeze(axis=1).str.split(expand=True), validate=False)),
    #('imputer', SimpleImputer(strategy='constant',fill_value=''))
])


# apply the pipeline of transformation for the tags
y_prog = tags_pipeline.fit_transform(df_prog.Tags)

# Use MultiLabelBinarizer to encode the tags
mlb = MultiLabelBinarizer()
y2 = mlb.fit_transform(y_prog[0])

# transform the title using the pipeline
title_proc = body_pipeline.fit_transform(df_prog.Title.to_frame())


tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False, stop_words=[''])
X_tfidf = tfidf_vectorizer.fit_transform(title_proc)

X_train, y_train, X_test, y_test = iterative_train_test_split(X_tfidf, y2, test_size = 0.2)

model = ClassifierChain(RandomForestClassifier(min_samples_split=15,n_estimators=10,
                                               n_jobs=4),order='random', random_state=1)

model.fit(X_train,y_train)

import dill

joblib.dump(model, 'C:/Users/elodi/Documents/model.pkl')



with open('C:/Users/elodi/Documents/model.pkl', 'wb') as pickle_file :
    dill.dump(model, pickle_file)


with open('C:/Users/elodi/Documents/pipeline.pkl', 'wb') as pickle_file :
    dill.dump(body_pipeline, pickle_file)

with open('C:/Users/elodi/Documents/vectorizer.pkl', 'wb') as vect_file:
    dill.dump(tfidf_vectorizer,vect_file)
    
with open('C:/Users/elodi/Documents/mlb.pkl', 'wb') as mlb_file:
    dill.dump(mlb,mlb_file)
    
    
model_load = joblib.load('C:/Users/elodi/Documents/model.pkl')

with open('C:/Users/elodi/Documents/pipeline.pkl', 'rb') as pickle_file :
    pipeline_load  = dill.load( pickle_file)
    

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:22:55 2021

@author: elodi
"""

# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import dill
import string
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

# Your API definition
app = Flask(__name__)


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
    stop = stopwords.words('English')
    return pd.Series([word for word in sentence[0].split() if word not in stop])


def remove_stopwords_df(df):
    return df.apply(remove_stopwords_sentence, axis=1)


def stem_sentence(s):
    stemmer = EnglishStemmer()
    return pd.Series([stemmer.stem(w) for w in s if not pd.isna(w)])



@app.route('/predict/', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            df_query = pd.DataFrame(json_)
            #print(df_query)
            
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
                        
            
            df_proc = body_pipeline.fit_transform(df_query.Title.to_frame())
            print(type(df_proc))
            #df_proc = pipeline.transform(df_query.Title.to_frame())
            tfidf = vectorizer.transform(df_proc)
            print(type(tfidf))
            prediction = model.predict(tfidf)
            
            prediction_tags = list(mlb.inverse_transform(prediction))

            return jsonify({'prediction': str(prediction_tags)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':

    #model = joblib.load('C:/Users/elodi/Documents/model.pkl')
    print('begin')
    with open('C:/Users/elodi/Documents/model.pkl', 'rb') as model_file :
        model  = dill.load( model_file)
    print ('Model loaded')
    
    with open('C:/Users/elodi/Documents/pipeline.pkl', 'rb') as pipeline_file :
        pipeline  = dill.load( pipeline_file)
    
    with open('C:/Users/elodi/Documents/vectorizer.pkl', 'rb') as vect_file :
        vectorizer  = dill.load( vect_file)
        
    with open('C:/Users/elodi/Documents/mlb.pkl', 'rb') as mlb_file :
        mlb  = dill.load( mlb_file)
    
    app.run(debug=True)
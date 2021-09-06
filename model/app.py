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
    """check if the word is a tag from the tags list and return its simplified form"""
    for t in list_tags:
        if word == t:
            simple_t = re.match('(\.?[a-z]+)', t).group()
            return (True, simple_t)
    return (False, '')


def simplify_tags_sentence(s, list_tags):
    """simplify a sentence of tags removing tags that dont belong to list_tags"""
    l = []
    for w in s.split(' '):
        b, x = check_tag(w, list_tags)
        if b:
            l.append(x)
    return l

def remove_stopwords_sentence(sentence):
    """remove stopwords from sentence"""
    stop = stopwords.words('English')
    return pd.Series([word for word in sentence[0].split() if word not in stop])


def remove_stopwords_df(df):
    """remove stopwords from a dataframe"""
    return df.apply(remove_stopwords_sentence, axis=1)


def stem_sentence(s):
    """ stem sentence"""
    stemmer = EnglishStemmer()
    return pd.Series([stemmer.stem(w) for w in s if not pd.isna(w)])



@app.route('/predict/', methods=['POST'])
def predict():
    if model:
        try:
            # get the request
            json_ = request.json
            
            # tranform to a datafram
            df_query = pd.DataFrame(json_)
           
            
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
               ])
                        
            # apply the preprocessor on the request data
            df_proc = body_pipeline.fit_transform(df_query.Title.to_frame())
           
            
            # apply the TF IDF vectorizer
            tfidf = vectorizer.transform(df_proc)
            
            # get the preiction from the model
            prediction = model.predict(tfidf)
            
            # transform the binary predictions to labels
            prediction_tags = list(mlb.inverse_transform(prediction))
            
            # send the response
            return jsonify({'prediction': str(prediction_tags)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':

    # load the model
    with open('C:/Users/elodi/Documents/model.pkl', 'rb') as model_file :
        model  = dill.load( model_file)
    print ('Model loaded')
    
    #with open('C:/Users/elodi/Documents/pipeline.pkl', 'rb') as pipeline_file :
        #pipeline  = dill.load( pipeline_file)
    # load the tfidf vectorizer
    with open('C:/Users/elodi/Documents/vectorizer.pkl', 'rb') as vect_file :
        vectorizer  = dill.load( vect_file)
    # load the MUltiLabelBinarizer    
    with open('C:/Users/elodi/Documents/mlb.pkl', 'rb') as mlb_file :
        mlb  = dill.load( mlb_file)
    
    app.run(debug=True)

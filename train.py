from sklearn import metrics, model_selection

import re
import sklearn
import nltk
import pandas as pd
import numpy as np
import xgboost as xgb
import spacy

import datetime

import importlib
import utility
import variation
import feature

from utility import *
from variation import Variation, Variations
from feature import *

# get training data for stage 2
train, y = get_stage2_train_data(preprocessed=False)
# make labels starts from 0
y = y - 1

test_all = get_stage2_test_data()
# consider only 150 real case, just don't want to waste time on fake ones
test = test_all[(test_all['real']==True)&(test_all['in_stage1']==False)]

# load spacy english model, this one performs better than the regular model
nlp = spacy.load('en_core_web_md')

# parse the variation, get information about
# site and amino acid
train['var'] = train.apply(lambda r: Variation(r['Variation']), axis=1)
test['var'] = test.apply(lambda r: Variation(r['Variation']), axis=1)

# this will do text cleaning and find sentences containing variation
get_text_features(train, nlp)
get_text_features(test, nlp)

# concat train and test, we will process them together
df_all = pd.concat((train, test), axis=0).drop(['Unnamed: 0', 'in_stage1', 'real', 'real_pred'], axis=1)
# this one will be used to calculate features based on former observations
df_ans = pd.concat((train, y), axis=1)

# is gene in text & is variation in text
df_share = get_share(df_all)
# gene onehot encoding
df_gene_onehot = get_gene_onehot(df_all)
# gene onehot encoding -> SVD (20 dimensions)
df_gene_svd = get_gene_svd(df_all)
# class distribution per gene
df_gene_dist = get_gene_dist(df_all, df_ans)
# encode variations by heuristic, extract features like
# amino acid, site, mutation type
df_dummy = get_var_dummy(df_all)
# get pre-trained Doc2Vec vectors
df_d2v_kuo = get_d2v_kuo()

def get_dist_count(df):
    df_dist = df.apply(lambda r: r/sum(r) if sum(r)>0 else r, axis=1)
    df_count = pd.DataFrame()
    df_count['count'] = df.apply(lambda r: sum(r), axis=1)
    # we can't really distinguish class 1/4 and 2/7,
    # maybe this will help to seperate 14/27
    df_count['count_14'] = df_dist.apply(lambda r: r[0] + r[3], axis=1)
    df_count['count_27'] = df_dist.apply(lambda r: r[1] + r[6], axis=1)
    
    return df_dist, df_count

# get neighbor_class feature, this is based on former observations
df_nc = get_neighbor_classes(df_all, df_ans)
df_nc_dist, df_nc_count = get_dist_count(df_nc)

# get point substitution type distribution
df_hc2 = get_hotspot_count(df_all, df_ans)
df_hc2_dist, df_hc2_count = get_dist_count(df_hc2)

# get text class distribution
df_tc = get_text_classes(df_all, df_ans)
df_tc_dist, df_tc_count = get_dist_count(df_tc)

# get tfidf features, either train a new model or
# transform from existing model
df_tfidf = get_tfidf(df_all, model_file='models/tfidf_all.pkl')
df_var_tfidf = get_var_tfidf(df_all, model_file='models/tfidf_var_all.pkl')

# number of sentences that contains the variation
var_len = df_all.apply(lambda r: len(r['Var_sent'].split(' . ')), axis=1).values.reshape(-1, 1)

# combining keywords
keywords = np.load('features/keyword_heuristic_array.npy')
new_words = ['increasing', 'binding', 'significantly', 'reduced',
             'increase', 'activat', 'signal', 'transform', 'enhance',
             'reduction', 'significant', 'decreased', 'may', 'consistent', 
             'loss of', 'slightly', 'greatly']
keywords_new = np.concatenate((keywords, new_words))

def word_freq(row, word):
    split = row['Var_sent'].split()
    if len(split) == 0:
        return 0
    else:
        return row['Var_sent'].count(word)/len(split)

# get keyword frequency per text
df_kf = df_all.apply(
    lambda r: pd.Series([word_freq(r, keyword) for keyword in keywords_new]), 
    axis=1)

# some important dependency bigram
dep_bigram = {'not show', 'not affect', 'not clear', 'not significant',
              'activity increase', 'significantly reduce'}
def get_dep_bigram_freq(row):
    split = row['Var_sent'].split()
    bigram_count = {key:0 for key in dep_bigram}
    if len(split) == 0:
        return bigram_count
    
    if row['ID'] % 500 == 0:
        print(row['ID'], end='\r')
    
    doc = nlp(row['Var_sent'], entity=False)
    for token in doc:
        if token.pos_ == 'PUNCT' or token.pos_ == 'DET':
            continue
        if token.lemma_ == token.head.lemma_:
            continue
        bigram = token.lemma_ + ' ' + token.head.lemma_
        if bigram in dep_bigram: 
            bigram_count[bigram] = bigram_count[bigram] + 1
    
    bigram_freq = {key:bigram_count[key]/len(split) for key in bigram_count}
    return bigram_freq
# get dependency bigram frequency
df_dep_bigram_freq = df_all.apply(lambda r: pd.Series(get_dep_bigram_freq(r)), axis=1)

# get pre-trained word vectors obtained from
# http://bio.nlplab.org/ for each gene
df_w2v_gene = get_w2v_gene()

# concatenate features
feat_all = np.hstack([df_w2v_gene, df_dummy, df_hc2_dist, df_hc2_count, 
                      df_tc_dist, df_tc_count, df_tfidf, var_len, 
                      df_kf, df_dep_bigram_freq])
# split training and testing set
feat_train = feat_all[:len(train)]
feat_test = feat_all[len(train):]
print(feat_train.shape)
print(feat_test.shape)


fold = 5
preds = []

# we'll run 5-fold cross-validation 3 times
for shift in [0, 100, 200]:
    scores = []
    
    for i in range(fold):
        params = {
            'eta': 0.02,
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': 9,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': i,
            'silent': True
        }
        """
        feat_train_idx = np.arange(feat_train.shape[0])
        id1, id2, y1, y2 = model_selection.train_test_split(feat_train_idx, y, test_size=0.15, random_state=i+shift)

        df_train1 = df_ans.iloc[id1]
        gd = get_gene_dist(df_train, df_train1).values
        
        x1 = np.hstack((feat_train[id1], gd[id1]))
        x2 = np.hstack((feat_train[id2], gd[id2]))
        """
        x1, x2, y1, y2 = model_selection.train_test_split(feat_train, y, test_size=0.15, random_state=i+shift)
        
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=50)
        score = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2)), labels = list(range(9)))
        print(score)
        scores.append(score)

        pred_test = model.predict(xgb.DMatrix(feat_test))
        preds.append(pred_test)

    print('\n', np.mean(scores))

# write submission file
write_submit_file('result/cv6815.csv', np.mean(preds, axis=0))

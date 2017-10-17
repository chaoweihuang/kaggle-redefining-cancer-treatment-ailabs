import re
import sklearn
import nltk
import pandas as pd
import numpy as np
import xgboost as xgb
import spacy
import datetime

from sklearn import metrics, model_selection

import utility
import variation
import feature

from utility import *
from variation import Variation, Variations
from feature import *
from model import Model


#######################################
#
# Preprocessing phase
#
# we will do preprocessing here.
# 1. get training data and testing data
# 2. text preprocessing
# 3. variation preprocessing
#
#######################################

# get training data for stage 2
train, y = get_stage2_train_data(preprocessed=False)
# make labels starts from 0
y = y - 1

test_all = get_stage2_test_data()
# consider only 150 real case
test = test_all[(test_all['real']==True)&(test_all['in_stage1']==False)]

try:
    # load spacy english model, this one performs better than the regular model
    nlp = spacy.load('en_core_web_md')
except:
    nlp = spacy.load('en')

if not nlp.parser:
    print('We need spacy parsers, please install spacy models first')
    print('Recommend model: en_core_web_md')
    exit(1)

# parse the variation, get information about
# site and amino acid
train['var'] = train.apply(lambda r: Variation(r['Variation']), axis=1)
test['var'] = test.apply(lambda r: Variation(r['Variation']), axis=1)

# this will do text cleaning and find sentences containing variation
get_text_features(train, nlp)
get_text_features(test, nlp)

# concat train and test, we will process them together
df_all = pd.concat((train, test), axis=0).drop(['Unnamed: 0', 'in_stage1', 'real', 'real_pred'], axis=1)
# this dataframe will be used to calculate features based on former observations
df_ans = pd.concat((train, y), axis=1)


#######################################
#
# Feature extracting phase
#
# we will do feature extraction here.
# 1. get all features (explanation
#    and extracting steps can be found
#    in feature.py)
#
#######################################

def get_dist_count(df):
    """
    Given vectors of counts, transform them
    into distribution and count for class
    14/27.
    """
    df_dist = df.apply(lambda r: r/sum(r) if sum(r)>0 else r, axis=1)
    df_count = pd.DataFrame()
    df_count['count'] = df.apply(lambda r: sum(r), axis=1)
    # we can't really distinguish class 1/4 and 2/7,
    # maybe this will help to seperate 14/27
    df_count['count_14'] = df_dist.apply(lambda r: r[0] + r[3], axis=1)
    df_count['count_27'] = df_dist.apply(lambda r: r[1] + r[6], axis=1)
    
    return df_dist, df_count

# is gene in text + is variation in text
df_share = get_share(df_all)

# encode variations by heuristic, extract features like
# amino acid, site, mutation type
df_dummy = get_var_dummy(df_all)
# get pre-trained Doc2Vec vectors
df_d2v_kuo = get_d2v_kuo()

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

# get keyword frequencies
df_kf_kuo = get_keyfreq_kuo()

# some important dependency bigram
dep_bigram = {'not show', 'not affect', 'not clear', 'not significant',
              'activity increase', 'significantly reduce'}

def get_dep_bigram_freq(row):
    """
    Count dependency bigram from Var_sent
    """
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
# we have dumped the vectors since the 
# size of the model is too large
df_w2v_gene = get_w2v_gene()

# get word vectors trained on training text
df_w2v_alex = get_w2v_alex(dim=100)


#######################################
#
# Training phase
#
# we will train 4 xgb models here.
# 1. combine features needed by each model
# 2. do cross validation, get the
#    average prediction of each fold
#
#######################################

fold = 5
repeat = 3

############### Model 1 ###############
feat_all = np.hstack([df_w2v_gene, df_dummy, df_hc2_dist, df_hc2_count, 
                      df_tc_dist, df_tc_count, df_tfidf, var_len, 
                      df_kf_kuo, df_dep_bigram_freq])
print(feat_all.shape)

feat_train = feat_all[:len(train)]
feat_test = feat_all[len(train):]

model_1 = Model()

preds_1 = model_1.cv(
            feat_train, y, feat_test,
            fold=fold, repeat=repeat
          )
preds_1 = np.mean(preds_1, axis=0)
############### Model 1 ###############

############### Model 2 ###############
feat_all = np.hstack([df_w2v_gene, df_dummy, df_hc2_dist, df_hc2_count,
                      df_tfidf, var_len, df_kf_kuo, df_dep_bigram_freq])

print(feat_all.shape)
feat_train = feat_all[:len(train)]
feat_test = feat_all[len(train):]

model_2 = Model()

preds_2 = model_2.cv(
            feat_train, y, feat_test,
            fold=fold, repeat=repeat
          )
preds_2 = np.mean(preds_2, axis=0)
############### Model 2 ###############

############### Model 3 ###############
feat_all = np.hstack([df_w2v_gene, df_w2v_alex, df_dummy,
                      df_hc2_dist, df_hc2_count, df_tfidf,
                      df_nc_dist, df_nc_count, df_kf_kuo
                     ])

print(feat_all.shape)
feat_train = feat_all[:len(train)]
feat_test = feat_all[len(train):]

model_3 = Model()

preds_3 = model_3.cv(
            feat_train, y, feat_test,
            fold=fold, repeat=repeat
          )
preds_3 = np.mean(preds_3, axis=0)
############### Model 3 ###############

############### Model 4 ###############
feat_all = np.hstack([df_w2v_gene, df_w2v_alex, 
                      df_hc2_dist, df_hc2_count, df_tfidf,
                      df_nc_dist, df_nc_count, df_kf_kuo
                     ])

print(feat_all.shape)
feat_train = feat_all[:len(train)]
feat_test = feat_all[len(train):]

model_4 = Model()

preds_4 = model_4.cv(
            feat_train, y, feat_test,
            fold=fold, repeat=repeat
          )
preds_4 = np.mean(preds_4, axis=0)
############### Model 4 ###############


#######################################
#
# Ensembling phase
#
# we will ensemble our models here.
# 1. do model switching
# 2. do model averaging
# 3. write submission files
#
#######################################

# do model switching
# if "text class distribution" feature is present: Model 1
# else if not present: Model 2
pred_switch = []
test_tc_count = np.array(df_tc_count['count'])
for pred_tc, pred_notc, count in zip(preds_1, preds_2, test_tc_count):
    if count == 0:
        pred_switch.append(pred_notc)
    else:
        pred_switch.append(pred_tc)

# do model averaging
# simply take means of predictions
pred_average = np.mean([preds_1, preds_2, preds_3, preds_4], axis=0)

# write submission file
# the submission using model switching is placed at 'results/switching.csv'
# the submission using model averaging is placed at 'results/averaging.csv'
write_submit_file('results/switching.csv', pred_switch)
write_submit_file('results/averaging.csv', pred_average)

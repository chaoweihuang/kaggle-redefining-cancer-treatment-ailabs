import os
import sys

import numpy as np
import pandas as pd
import nltk 
import re
import sklearn
import spacy

from sklearn.externals import joblib
from sklearn import decomposition, feature_extraction, pipeline

from variation import Variation, Variations
from utility import *


def get_share(df):
    def _get_share(df):
        df_share = pd.DataFrame()
        df_share['gene_share'] = df.apply(lambda r: r['Gene'].lower() in r['Text'].lower(), axis=1)
        df_share['variation_share'] = df.apply(lambda r: r['Variation'].lower() in r['Text'].lower(), axis=1)
        return df_share

    df_share = _get_share(df)

    return df_share

def get_gene_onehot(df):
    df_gene_onehot = pd.get_dummies(df['Gene'])

    return df_gene_onehot

def get_gene_svd(df):
    df_gene_onehot = pd.get_dummies(df['Gene'])

    svd = decomposition.TruncatedSVD(n_components=20, n_iter=20, random_state=12)
    df_gene_svd = pd.DataFrame(svd.fit_transform(df_gene_onehot.values))
    
    return df_gene_svd

def get_gene_dist(df, df_ans):
    def _get_gene_dist(row):
        gene_dist = np.zeros(9).astype(int)
        targets = df_ans[df_ans['Gene']==row['Gene']]
        for index, trow in targets.iterrows():
            #if trow['Variation'] == row['Variation']:
            #    continue
            gene_dist[trow['Class']] += 1

        return gene_dist
    
    def _get_gene_dist_df(df):
        gene_dist = []
        for i, (index, row) in enumerate(df.iterrows()):
            gd = _get_gene_dist(row)
            gene_dist.append(gd / gd.sum() if gd.sum()>0 else gd)
        
        df_gene_dist = pd.DataFrame(gene_dist)
        return df_gene_dist

    df_gene_dist = _get_gene_dist_df(df)

    return df_gene_dist

def get_hotspot_count(df, df_ans):
    def _get_hotspot_count(row):
        hotspot_count = np.zeros(9).astype(int)
        var = row['Variation']
        Var = row['var']
        #if (Var.start_amino == '') or (Var.pos == 0 and Var.start_pos == 0):
        if (Var.start_amino == '') or (Var.pos == 0):
            return hotspot_count
        
        #pos = Var.pos if Var.pos != 0 else Var.start_pos
        pos = Var.pos
        hotspot = '{}{}'.format(Var.start_amino, str(pos)).lower()

        targets = df_ans[df_ans['Gene']==row['Gene']]
        print(var, hotspot)
        for index, trow in targets.iterrows():
            tvar = trow['Variation']
            if len(tvar) > 6: continue
            if tvar.lower().startswith(hotspot) and not (tvar == var):
                print('\t', tvar, trow['Class'])
                hotspot_count[trow['Class']] += 1

        return hotspot_count
    
    def _get_hotspot_count_df(df):
        hotspot_count = []
        for index, row in df.iterrows():
            hc = _get_hotspot_count(row)
            hotspot_count.append(hc)
        
        df_hotspot_count = pd.DataFrame(hotspot_count)
        return df_hotspot_count

    df_hc = _get_hotspot_count_df(df)
    
    return df_hc

def get_var_dummy(df):
    df_dummy = Variations(df['Variation']).to_dummy_df().astype(int)
    
    return df_dummy

def get_neighbor_classes(df, df_ans):
    def cleanup(x):
        import string
        return x.translate(str.maketrans("", "", string.punctuation.replace('*', '')))

    def _get_neighbor_classes(row):
        neighbor_classes = np.zeros(9).astype(int)
        text = row['Text'].lower()
        var = row['Variation'].lower()
        Var = row['var']
        gene = row['Gene']
        targets = df_ans[df_ans['Gene']==gene]
        
        for sent in row['Var_sent'].split(' . '):
            for index, trow in targets.iterrows():
                tvar = trow['Variation'].lower()
                if (trow['var'].find_sent(sent, exact=True)) and (not tvar == var):
                    neighbor_classes[trow['Class']] += 1

        return neighbor_classes

    def _get_neighbor_classes_df(df):
        neighbor_classes = []
        for i, (index, row) in enumerate(df.iterrows()):
            if i % 500 == 0: 
                print('[INFO] processed {} rows'.format(i))
            nc = _get_neighbor_classes(row)
            neighbor_classes.append(nc)
        
        df_neighbor_classes = pd.DataFrame(neighbor_classes)
        return df_neighbor_classes

    df_nc = _get_neighbor_classes_df(df)

    return df_nc

def get_text_classes(df, df_ans):
    def _get_text_classes(row):
        text_classes = np.zeros(9).astype(int)
        targets = df_ans[df_ans['Gene']==row['Gene']]
        
        for index, trow in targets.iterrows():
            i = 0
            while i+200 <= len(row['Text']):
                sent = row['Text'][i:i+200]

                if (sent in trow['Text']) and (not trow['Variation'] == row['Variation']):
                    text_classes[trow['Class']] += 1
                    break
                i += 3000

        return text_classes

    def _get_text_classes_df(df):
        text_classes = []
        for i, (index, row) in enumerate(df.iterrows()):
            if i % 500 == 0: 
                print('[INFO] processed {} rows'.format(i))
            tc = _get_text_classes(row)
            text_classes.append(tc)
        
        df_text_classes = pd.DataFrame(text_classes)
        return df_text_classes

    df_tc = _get_text_classes_df(df)

    return df_tc

def get_d2v_kuo():
    train_d2v = np.load('features/feat_doc2vec_train.npy')
    test_d2v = np.load('features/feat_doc2vec_test.npy')

    df_d2v_kuo = pd.DataFrame(np.vstack((train_d2v, test_d2v)))
    return df_d2v_kuo

def get_keyfreq_kuo():
    train_kf = np.load('features/feat_keyfreq_train.npy')
    test_kf = np.load('features/feat_keyfreq_test.npy')

    df_kf_kuo = pd.DataFrame(np.vstack((train_kf, test_kf)))
    return df_kf_kuo

def get_w2v_alex(dim=300):
    train_w2v = np.load('features/train_stage2_dim{}.npy'.format(str(dim)))
    test_w2v = np.load('features/test_stage2_dim{}.npy'.format(str(dim)))
    
    df_w2v = pd.DataFrame(np.vstack((train_w2v, test_w2v)))
    return df_w2v

def get_w2v_gene():
    vectors = np.load('features/gene_vectors.npy')
    df_w2v = pd.DataFrame(vectors)

    return df_w2v

def get_feat_from_npy(f):
    feat = np.load(f)
    df_feat = pd.DataFrame(feat)

    return df_feat

def clean_text(text, nlp):
    doc = nlp(text, parse=False, entity=False)
    words = [token.text for token in doc if not token.pos_=='PUNCT']
    return ' '.join(words)

def clean_texts(texts, nlp):
    docs = []
    for text in texts:
        docs.append(clean_text(text, nlp))
    return docs

def get_text_features(df, nlp=None):
    if nlp == None:
        nlp = spacy.load('en_core_web_md')
    
    if not 'Text_clean' in df.columns:
        print('[INFO] cleaning text')
        df['Text_clean'] = df.apply(lambda r: clean_text(r['Text'], nlp).lower(), axis=1)
    if not 'Var_sent' in df.columns:
        print('[INFO] finding variation sentences')
        df['Var_sent'] = df.apply(lambda r: ' . '.join(Variation(r['Variation']).find_sents(r['Text'], exact=False)), axis=1)
    if not 'Var_sent_clean' in df.columns:
        print('[INFO] cleaning variation sentences')
        df['Var_sent_clean'] = df.apply(lambda r: clean_text(r['Var_sent'], nlp).lower(), axis=1)
    
    return

def get_tfidf(df, model_file=None, params=None):
    if params == None:
        params = {}

    text_clean = df['Text_clean']

    if model_file == None or not os.path.exists(model_file):
        model_tfidf = feature_extraction.text.TfidfVectorizer(
            ngram_range=(1, 2), stop_words=None, 
            max_features=200000, max_df=0.8, min_df=5, **params
        )
        model_svd = decomposition.TruncatedSVD(
            n_components=200, n_iter=25, random_state=12
        )

        model = pipeline.make_pipeline(model_tfidf, model_svd)
        print('[INFO] fitting tfidf model')
        model.fit(text_clean)
        if model_file is not None:
            joblib.dump(model, model_file)
    else:
        print('[INFO] loading tfidf model from {}'.format(model_file))
        model = joblib.load(model_file)

    print('[INFO] transforming tfidf')
    df_tfidf = pd.DataFrame(model.transform(text_clean))

    return df_tfidf

def get_var_tfidf(df, model_file=None, params=None):
    if params == None:
        params = {}

    var_text_clean = df['Var_sent_clean']

    if model_file == None or not os.path.exists(model_file):
        model_tfidf = feature_extraction.text.TfidfVectorizer(
            ngram_range=(1, 2), stop_words=None, 
            max_features=50000, max_df=0.3, min_df=2, **params
        )
        model_svd = decomposition.TruncatedSVD(
            n_components=30, n_iter=25, random_state=12
        )

        model = pipeline.make_pipeline(model_tfidf, model_svd)
        print('[INFO] fitting var tfidf model')
        model.fit(var_text_clean)
        if model_file is not None:
            joblib.dump(model, model_file)
    else:
        print('[INFO] loading tfidf model from {}'.format(model_file))
        model = joblib.load(model_file)

    print('[INFO] transforming var tfidf')
    df_var_tfidf = pd.DataFrame(model.transform(var_text_clean))

    return df_var_tfidf


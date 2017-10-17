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
    """
    Given a dataframe, extract 2 features:
    1. is gene name present in text
    2. is variation name present in text
    """
    def _get_share(df):
        df_share = pd.DataFrame()
        df_share['gene_share'] = df.apply(lambda r: r['Gene'].lower() in r['Text'].lower(), axis=1)
        df_share['variation_share'] = df.apply(lambda r: r['Variation'].lower() in r['Text'].lower(), axis=1)
        return df_share

    df_share = _get_share(df)

    return df_share

def get_gene_onehot(df):
    """
    Given a dataframe, get one-hot encoding of gene names
    """
    df_gene_onehot = pd.get_dummies(df['Gene'])

    return df_gene_onehot

def get_gene_svd(df):
    """
    Given a dataframe, get one-hot encoded feature
    and apply SVD on them
    """
    df_gene_onehot = pd.get_dummies(df['Gene'])

    svd = decomposition.TruncatedSVD(n_components=20, n_iter=20, random_state=12)
    df_gene_svd = pd.DataFrame(svd.fit_transform(df_gene_onehot.values))
    
    return df_gene_svd

def get_gene_dist(df, df_ans):
    """
    Given a dataframe, get class distributions for each gene
    and use as features for each data point
    """
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
    """
    For every point substitution (e.g., M224R), 
    we searched all the data points in the available 
    training set to find those with the same beginning
    amino acid and mutation site (in this case, M224).
    
    If some data points were found, the variation 
    class distribution was calculated accordingly;
    otherwise, this feature was treated as missing.
    
    For other variation formats, this feature was 
    treated as missing.
    """
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
    """
    We extract 2 features here: feature with index 2, 3
    in the description file

    1.  Variation formats, such as truncating mutations,
        ins (e.g., Q58_Q59insL), del (e.g., S459del),
        point substitution (e.g., M224R), were encoded
        using a one-hot aka one-of-K scheme. There are 
        28 variation formats encoded in total.

    2.  For every point substitution (e.g., M224R), 
        the beginning and ending amino acids (e.g., M and R)
        were encoded by one-hot (20 different possible 
        amino acids for each), and the mutation site (e.g., 224)
        was encoded numerically. 
        For other variation formats, this feature was 
        treated as missing.
    """
    df_dummy = Variations(df['Variation']).to_dummy_df().astype(int)
    
    return df_dummy

def get_neighbor_classes(df, df_ans):
    """
    For every data point, we found the paragraphs
    that contain the associated variation. 

    All other variations in these paragraphs are 
    called neighbors of the data point. If some 
    neighbors were found, their class distribution
    was calculated according to the corresponding labels;
    otherwise, this feature was treated as missing.
    """
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
    """
    For every data point, we computed “class distribution of the text."
    Specifically, several segments (each with 200 characters) of the 
    text were extracted. 
    
    We then used the extracted segments to search all the data points
    in the available training set to find those with the same segments.
    If some data points were found, the class distribution was 
    calculated accordingly; otherwise, this feature was treated as missing.
    (We handled this issue in the Model Switching/Averaging section.)
    """
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
    """
    Load the doc2vec features trained on training texts. 
    We already extracted this feature in pretrain stage

    Feature description:
    Vector representations of texts (doc2vec) were induced 
    from the paragraphs containing the variation name by gensim.
    (For every data point whose variation name cannot be found 
    in its text, 0-vector was adopted.)
    """
    train_d2v = np.load('features/feat_doc2vec_train2.npy')
    test_d2v = np.load('features/feat_doc2vec_test2.npy')

    df_d2v_kuo = pd.DataFrame(np.vstack((train_d2v, test_d2v)))
    return df_d2v_kuo

def get_keyfreq_kuo():
    """
    Load the produced keyword frequency features. We already
    extracted this feature in pretrain stage

    Feature description:
    Texts were grouped according to their labels into 9 meta-texts.
    Keywords were then identified by tf-idf of the 9 meta-texts.
    (Some keywords were added heuristically.) 
    
    For every data point, frequencies of these keywords (and the 
    negation of them) in the text were calculated. For example, 
    “likely” was identified as a keyword from tf-idf, then we 
    computed the frequencies of “likely” and “not likely” in the 
    given text.
    """
    train_kf = np.load('features/feat_keyfreq_train2.npy')
    test_kf = np.load('features/feat_keyfreq_test2.npy')

    df_kf_kuo = pd.DataFrame(np.vstack((train_kf, test_kf)))
    return df_kf_kuo

def get_w2v_alex(dim=300):
    """
    Load the word vectors trained on training text. Note that we
    already induced the vectors from our FastText model because
    the size of model is too large.

    Feature description:
    Vector representations of variations (word2vec) were induced
    from the texts of all data points (the training and testing 
    datasets of Stage 2) by fastText.
    """
    train_w2v = np.load('features/train_stage2_dim{}.npy'.format(str(dim)))
    test_w2v = np.load('features/test_stage2_dim{}.npy'.format(str(dim)))
    
    df_w2v = pd.DataFrame(np.vstack((train_w2v, test_w2v)))
    return df_w2v

def get_w2v_gene():
    """
    Load the pre-trained word vectors of gene names. Note that
    we already induced the word vectors since the size of model
    is too large. The original Word2Vec model can be found at
    http://bio.nlplab.org/#word-vectors

    Feature description:
    Vector representations of genes (word2vec) were induced from
    PubMed and PMC texts. (External data usage from biomedical 
    natural language processing http://bio.nlplab.org)
    """
    vectors = np.load('features/gene_vectors.npy')
    df_w2v = pd.DataFrame(vectors)

    return df_w2v

def get_feat_from_npy(f):
    feat = np.load(f)
    df_feat = pd.DataFrame(feat)

    return df_feat

def clean_text(text, nlp):
    """
    Do text cleaning.
    We remove punctuation in the text.
    """
    doc = nlp(text, parse=False, entity=False)
    words = [token.text for token in doc if not token.pos_=='PUNCT']
    return ' '.join(words)

def clean_texts(texts, nlp):
    """
    Clean all given texts
    """
    docs = []
    for text in texts:
        docs.append(clean_text(text, nlp))
    return docs

def get_text_features(df, nlp=None):
    """
    Preprocess text. We extract 3 features here:
    
    1. Text_clean: cleaned Text
    2. Var_sent: sentence that the associated variation presents
    3. Var_sent_clean: cleaned Var_sent
    """
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
    """
    TF-IDF from the texts (documents) of all data points
    (the training and testing datasets of Stage 2) was 
    computed and then SVD was used to reduce the number of 
    dimensions from 200k (terms) to 200 (principal components).
    """
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
    """
    TF-IDF from the texts (documents) of all Var_sents (see get_text_features).
    (the training and testing datasets of Stage 2) was 
    computed and then SVD was used to reduce the number of 
    dimensions from 200k (terms) to 200 (principal components).
    """
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


import numpy as np
import pandas as pd
import sklearn
import json
import re


def get_stage2_train_data(preprocessed=False):
    train = pd.read_csv('data/train_stage2.csv')
    y = train['Class']
    train = train.drop(['Class'], axis=1)

    return train, y

def get_stage2_test_data(preprocessed=False):
    test = pd.read_csv('data/stage2_test_tag.csv')

    return test

def train_test_split_by_gene(df, test_size=0.15, random_state=0):
    np.random.seed(random_state)
    
    genes = df.Gene.unique()
    np.random.shuffle(genes)
    
    index = np.zeros(len(df)).astype(bool)
    thres = len(df) * (1 - test_size)
    for gene in genes:
        index[df['Gene'] == gene] = True
        if np.sum(index) > thres:
            break
    
    test_index = np.logical_not(index)
    return index, test_index

def get_amino_alias(amino, full=False):
    data_file = 'data/one2many.json'
    d = json.load(open(data_file))
    
    if amino.upper() in d:
        if not full:
            return d[amino.upper()][-1]
        else:
            return d[amino.upper()][0]
    
    return amino

def findall(text, idx=0):
    alpha_list = set(list('abcdefghijklmnopqrstuvwxyz'))
    words = text.lower().split()
    found = []
    for i, word in enumerate(words):
        if not word[0] == '.':
            continue
        if not len(word)<20:
            continue
        if len(word) < 2:
            continue
        if len(word) == 2 and word[1] == 'a':
            continue
        if not word[1] in alpha_list:
            continue

        sent = ' '.join(words[i-4:i+5])
        flag = False
        for sp in ['.', ';', '(', ')', ':', '"', "'"]:
            if sent.count(sp) > 3:
                flag = True
                break
        if flag:
            continue

        flag = False
        for sp in ['..', '--', ';;', '::', '""', "''"]:
            if sp in sent:
                flag = True
                break
        if flag:
            continue

        found.append(word)

    return found

def get_fake_feature(df):
    dot_count = df.apply(lambda row: len(findall(row['Text'], row['ID'])), axis=1)
    var_count = df.apply(lambda row: row['Text'].lower().count(row['Variation'].lower()), axis=1)
    var_freq = df.apply(lambda row: row['Text'].lower().count(row['Variation'].lower())/len(row['Text'].split()), axis=1)
    gene_appear = df.apply(lambda row: row['Gene'].lower() in row['Text'].lower(), axis=1)
    word_count = df.apply(lambda row: len(row['Text'].split()), axis=1)
    return pd.concat([dot_count, var_count, var_freq, gene_appear, word_count], axis=1)
    
def write_submit_file(f, pred):
    df_ans = pd.read_csv('data/sample_submission.csv')
    real_id = np.array(pd.read_csv('data/stage2_test_real.csv')['ID'])
    df_pred = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    df_pred['ID'] = real_id
    for index, row in df_pred.iterrows():
        df_ans.iloc[int(row['ID'])-1] = row

    df_ans['ID'] = df_ans['ID'].astype(int)
    df_ans.to_csv(f, index=False)

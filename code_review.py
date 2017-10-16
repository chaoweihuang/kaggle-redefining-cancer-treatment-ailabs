
# coding: utf-8

# FEATURE: keyword frequency
# PLEASE ADD: the path of tfidf_by_class_adj or adv files
key_adj = pd.read_csv('tfidf_by_class_adj.csv', error_bad_lines=False)[['class','tfidf','word']]
key_adv = pd.read_csv('tfidf_by_class_adv.csv', error_bad_lines=False)[['class','tfidf','word']]

# filter by tfidf value
threshold = 0.18
keywords_adj = key_adj[key_adj['tfidf'] >= threshold]
keywords_adv = key_adv[key_adv['tfidf'] >= threshold]
keywords = pd.concat([keywords_adj,keywords_adv], axis=0).reset_index(drop=True)
keywords_list = list(keywords['word'].unique())

# add heuristic keywords
keywords_heuristic = ['gain', 'loss', 'switch', 'homozygous', 'neutral', 'exon',
                      'functional', 'oncogenic', 'resistant', 'recurrent', 'vus',
                      'deleterious', 'nick', 'variant', 'primary', 'sensitive',
                      'neratinib', 'transgenic', 'truncated', 'aberrant','hematopoietic',
                      'alternative', 'very', 'still','predominantly', 'probably','transiently',
                      'alone', 'potentially','functionally', 'fully', 'effectively', 'possibly',
                      'likely','largely', 'importantly', 'stably','constitutively', 'poorly',
                      'differentially', 'directly']
for word in keywords_heuristic:
    keywords_list.append(word)

# add negations of keywords
keywords_neg = []
for word in keywords_list:
    keywords_neg.append('not '+word)
for neg_word in keywords_neg:
    keywords_list.append(neg_word)

# count keyword frequency for variation paragraph
import re
def count_key_freq_paragraph(sen, keyword_list):
    '''
    for a given sentence where variation name appears,
    return frequency of each keyword on the keyword_list,
    in the form of a numpy array
    '''
    vec = np.zeros(len(keyword_list))
    for j in range(vec.shape[0]):
        pattern = keyword_list[j]
        vec[j] = len(re.findall(pattern, sen))
    vec /= len(sen)
    return vec

def get_feat_keyfrq_paragraph(table, keyword_list, pre=100, post=300):
    '''
    for a given table of training/testing dataset, and a keyword list,
    return the feature of keyword frequency,
    in the form of a numpy array
    '''
    feat_keyfrq = np.zeros([table.shape[0],len(keyword_list)])
    for i in range(feat_keyfrq.shape[0]):
        text = table['Text'][i]
        var = table['Variation'][i][:-1]
        gene = table['Gene'][i]
        paragraph = []
        for match_V in re.finditer(var, text):
            sentence = ''
            s = match_V.start()
            e = match_V.end()
            sentence = text[s-pre:e+post]
            paragraph.append(sentence)
        for match_G in re.finditer(gene, text):
            sentence = ''
            s = match_G.start()
            e = match_G.end()
            sentence = text[s-pre:e+post]
            paragraph.append(sentence)
        gv_sen = ''
        for sen in paragraph:
            gv_sen += sen
        feat_keyfrq[i] = count_key_freq_paragraph(gv_sen, keyword_list)
    return feat_keyfrq


# get keyword frequnecy feature for stage2 training and testing dataset
# PLEASE ADJUST: the name of dataframe of stage 2 training and testing dataset 
# for the following 2 statements
feat_keyfreq_train2 = get_feat_keyfrq_paragraph(train2, keywords_list)
feat_keyfreq_test2 = get_feat_keyfrq_paragraph(test2, keywords_list)

np.save('feat_keyfreq_train2.npy', feat_keyfreq_train2)
np.save('feat_keyfreq_test2.npy', feat_keyfreq_test2)

# FEATURE: d2v

# transform table to labelled doc
# variation only; no stemming; only manually de-pleural
import re
from collections import namedtuple

def table2lbldoc(table, pre_char, post_char):
    '''
    for each data point in the given table of training/testing dataset,
    and assigned window of pre-characters and posy-characters,
    find the paragraph where variation name appears in its text in the form of labeldoc.
    final return is a list of labeldoc for those successfully parsed data points, 
    and a list of indeces of dropouts for those whose variation name cannot be found in text.
    '''
    # heuristic stopwords
    stopwords = ['wa', 'table', 'fig','figure','1','2','3','4','5','6','7','8','9','at',
                 'and','were','to','or','with','','a','am','an','are','be','been','being',
                 'by','for','had','has','have','he','her','her','hers','him','his','i',
                 'in','is','me','my','mine','of','our','ours','own','she','the','their',
                 'them','themselves','there','us','was','we','who','whoever','whom',
                 'whose','you','your','yours','yourself','yourselves']
    
    LabelDoc = namedtuple('LabelDoc', 'words tags')
    all_docs = []
    count = 0
    count_drop_V = 0
    drop_list = []
    
    for i in range(table.shape[0]):
        paragraph_V = []
        tag = [] 
        text = table['Text'][i]
        pattern_V = table['Variation'][i][1:-1] # ignore the first and last char of variation name
        for match_V in re.finditer(pattern_V, text):
            sentence = ''
            s = match_V.start()
            e = match_V.end()
            sentence = text[s-pre_char:e+post_char]
            word_list = sentence.split(' ')[1:-1]
            # de-pleural
            for j, item in enumerate(word_list):
                if len(item) > 0:
                    if item[-1] == 's':
                        word_list[j] = item[:-1]
            paragraph_V.append(word_list)       
        # remove stop words
        paragraph_V = [item for items in paragraph_V for item in items if item not in stopwords]
        # count dropouts
        if len(paragraph_V) < 3:
            #print('Var', i, 'not in text, length: ', len(paragraph_V))
            count_drop_V += 1
            drop_list.append(i)
            continue
        else:
            tag.append(count)
            count += 1
            all_docs.append(LabelDoc(paragraph_V, tag))
    print('total number of drop by var: ', count_drop_V)
    return all_docs, drop_list



# build dataframe for train & true test: df_all_doc2vec
# PLEASE ADJUST: the variable name for stage 2 training and testing dataset
df_all_doc2vec = pd.concat([train2[['Gene', 'Variation', 'Text']], test2[['Gene', 'Variation', 'Text']]], axis=0).reset_index()

# transform all data to labelled doc
pre_char = 100
post_char = 300
lbldoc_all, drop_ind_all = table2lbldoc(df_all_doc2vec, pre_char, post_char)

# for reshuffling
doc_list = lbldoc_all[:]

# generate training drop list and testing drop list from drop_ind_all
# PLEASE ADJUST: the variable name for stage 2 training and testing dataset
drop_ind_train2 = [e for e in drop_ind_all if e < train2.shape[0]]
drop_ind_test2 = [e for e in drop_ind_all if e > train2.shape[0]-1]



# import gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
from random import shuffle

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

# build model
model = Doc2Vec(dm=1, size=100, window=5, negative=5, hs=0, min_count=2)
#buiild_vocab
model.build_vocab(lbldoc_all)

# train model: decrease learning rate and shuffling
loop = 30
for epoch in range(loop):
    #print(epoch, model.corpus_count, model.iter)
    shuffle(doc_list)  # Shuffling gets better results
    model.train(lbldoc_all, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

model.save('model.doc2vec')



# generate d2v vector for training and true testing dataset
doc2vec_raw = model.docvecs.doctag_syn0
# PLEASE ADJUST: the variable name for stage 2 training dataset
feat_doc2vec_train2 = np.zeros([train2.shape[0],doc2vec_raw.shape[1]])
count_take_train2 = 0
for i in range(train2.shape[0]):
    if i not in drop_ind_train2:
        feat_doc2vec_train2[i] = doc2vec_raw[count_take_train2]
        #print('train2 vector', i, 'from raw vector', count_take_train2)
        count_take_train2 += 1
# PLEASE ADJUST: the variable name for stage 2 testing dataset
feat_doc2vec_test2 = np.zeros([test2.shape[0],doc2vec_raw.shape[1]])
count_take_test2 = count_take_train2
for i in range(test2.shape[0]):
    if i+train2.shape[0] not in drop_ind_test2:
        feat_doc2vec_test2[i] = doc2vec_raw[count_take_test2]
        #print('test2 vector', i, '(', i+train2.shape[0], ') from raw vector', count_take_test2)
        count_take_test2 += 1

np.save('feat_doc2vec_train.npy', feat_doc2vec_train2)
np.save('feat_doc2vec_test.npy', feat_doc2vec_test2)


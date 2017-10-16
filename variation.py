import re

import numpy as np
import pandas as pd
import json
import nltk

from utility import get_amino_alias


class Variation(object):
    def __init__(self, var):
        self.var = var
        self.type = ''
        self.start_pos = 0
        self.end_pos = 0
        self.start_amino = ''
        self.end_amino = ''
        self.pos = 0
        self.stop_sign = False
        
        self.alias_dict = {'ins': 'insertion', 'dup': 'duplication'}
        self.exact_alias = []
        self.alias = []

        self.parse()

    def to_dict(self):
        return {'type': self.type, 'start_pos': self.start_pos,
                'end_pos': self.end_pos, 'start_amino': self.start_amino,
                'end_amino': self.end_amino, 'pos': self.pos,
                'stop_sign': self.stop_sign,
                'pos_diff': self.end_pos - self.start_pos} 

    def parse(self):
        """This is a naive parser
        using a lot of heuristics and observations
        """

        special_vars = {'amplification', 'copy number loss', 
            'epigenetic silencing', 'overexpression'}

        special_terms = ['dna binding domain', 'egfrv', 'truncating mutation',
                        'fusions', 'fusion', 'mutation', 'deletion', 
                        'duplication', 'insertion', 'hypermethylation']
        
        special_var_alias = {
            'amplification': ['amplification', 'amplif', 'amp'],
            'copy number loss': ['copy number loss', 'copy number'],
            'epigenetic silencing': ['epigenetic silencing', 'silencing',
                                     'silenc', 'epigenetic'],
            'overexpression': ['overexpression', 'expression']
        }
        special_term_alias = {
            'dna binding domain': ['dna binding', 'binding', 'dna'],
            'egfrv': ['egfr'],
            'truncating mutation': ['truncating mutation', 'truncating',
                                    'truncated', 'truncate'],
            'fusions': ['fusions', 'fusion'],
            'fusion': ['fusion'],
            'mutation': ['mutation', 'mutant', 'mutating', 'mutate'],
            'deletion': ['deletion', 'delet', 'del'],
            'duplication': ['duplication', 'duplicat', 'dup'],
            'insertion': ['insertion', 'insert', 'ins'],
            'hypermethylation': ['hypermethylation', 'hypermethy'],
            'del': ['delet', 'del'],
            'trunc': ['trunc'],
            'splice': ['splice', 'splic'],
            'fs': ['fs', 'frame shift']
        }

        var = self.var.lower()

        # Check if the stop sign '*' in the variation
        if '*' in var:
            self.stop_sign = True
        
        # Type "exact match with special pre-difined variations"
        if var in special_vars:
            self.type = var
            self.alias += special_var_alias.get(self.type, [])
            return
        
        # Type "with special term"
        for term in special_terms:
            if term in var:
                self.type = term
                self.alias += special_term_alias.get(self.type, [])
                return

        # Type "point": A123B or A123* or A123
        if re.match('^[a-z][0-9]+[a-z|*]?$', var):
            split = re.split('[0-9]+', var)
            self.type = 'point'
            self.start_amino = split[0]
            self.end_amino = split[1]
            s = re.search('[0-9]+', var)
            self.pos = int(s.group())
            self._get_exact_alias()
            self._get_alias()
            return

        # Type "del/ins/trunc/splice/dup/fs": A123del or A123_B234del
        for suffix in ['del', 'ins', 'trunc', 'splice', 'dup', 'fs']:
            if suffix in var:
                self.type = self.alias_dict.get(suffix, suffix)
                self._parse_suffix(var, suffix)
                self.alias += special_term_alias.get(self.type, [])
                return

        print('[INFO] variation cannot be parsed: %s' % self.var)

    def _parse_suffix(self, var, suffix):
        var_nosuffix = var.split(suffix)[0]
        if re.match('^[a-z]?[0-9]+[a-z]?[_]?$', var_nosuffix):
            # ex: T123del or T123_splice
            var_nosuffix = var_nosuffix.replace('_', '')
            self.start_amino, self.pos = self._parse_amino_pos(var_nosuffix)
        elif re.match('^[a-z]?[0-9]+[a-z]?_[a-z]?[0-9]+[a-z]?$', var_nosuffix):
            # ex: T123_A345del
            start, end = re.split('_', var_nosuffix)
            self.start_amino, self.start_pos = self._parse_amino_pos(start)
            self.end_amino, self.end_pos = self._parse_amino_pos(end)
        return

    def _parse_amino_pos(self, var):
        """This function is for parsing amino and pos
        from "t123" or "123" or "t123r"(last r is ignored)
        """
        if re.match('[a-z]', var[-1]):
            var = var[:-1]

        amino = ''
        if re.match('^[a-z]', var):
            amino = var[0]
            pos = int(var[1:])
        else:
            pos = int(var)

        return amino, pos

    def _get_exact_alias(self):
        if not self.type == 'point':
            return
        
        alias_start = get_amino_alias(self.start_amino)
        alias_end = get_amino_alias(self.end_amino)
        alias = '{}{}{}'.format(alias_start, str(self.pos), alias_end)
        self.exact_alias.append(alias)

        return

    def _get_alias(self):
        if not self.type == 'point':
            return
        
        alias_start = get_amino_alias(self.start_amino)
        alias_end = get_amino_alias(self.end_amino)
        
        puncs = [' ', ',', '.', ')', '/', ';']
        alias_set = {'{}{}{}'.format(self.start_amino, str(self.pos), punc) for punc in puncs}
        alias_set = alias_set.union({'{}{}{}'.format(alias_start, str(self.pos), punc) for punc in puncs})
        self.alias.append(alias_set)

        alias_set = {'{}-{}{}'.format(self.start_amino, str(self.pos), punc) for punc in puncs}
        alias_set = alias_set.union({'{}-{}{}'.format(alias_start, str(self.pos), punc) for punc in puncs})
        self.alias.append(alias_set)
        
        alias_set = {'{} {}{}'.format(alias_start, str(self.pos), punc) for punc in puncs}
        self.alias.append(alias_set)
        
        self.alias.append(['codon', '{}'.format(str(self.pos))])
        self.alias.append({'c.{}'.format(str(self.pos*3 - i)) for i in range(3)})
        self.alias.append(['{}'.format(get_amino_alias(self.start_amino, full=True)),
                           '{}'.format(get_amino_alias(self.end_amino, full=True)),
                           'to'])
        self.alias.append(['{}'.format(get_amino_alias(self.start_amino, full=False)),
                           '{}'.format(get_amino_alias(self.end_amino, full=False)),
                           'to'])
        self.alias.append(['{}'.format(get_amino_alias(self.start_amino, full=True)),
                           '{}'.format(get_amino_alias(self.end_amino, full=True))])
        self.alias.append(['{}'.format(get_amino_alias(self.start_amino, full=False)),
                           '{}'.format(get_amino_alias(self.end_amino, full=False))])
        #self.alias.append('{}'.format(str(self.pos)))

        return

    def find_sent(self, sent, exact=False):
        def _find_alias_recurse(alias, sent):
            if isinstance(alias, set):
                return any([_find_alias_recurse(item, sent) for item in alias])
            elif isinstance(alias, list):
                return all([_find_alias_recurse(item, sent) for item in alias])
            else:
                return alias in sent
       
        sent = sent.lower()

        sent = ' ' + sent + ' '
        find_exact = False
        if self.var.lower() in sent:
            find_exact = True
        else:
            for alias in self.exact_alias:
                if alias in sent:
                    find_exact = True
                    break

        if exact == True or find_exact:
            return find_exact
        
        find_alias = False
        for alias in self.alias:
            if _find_alias_recurse(alias, sent):
                find_alias = True
                break

        return find_alias

    def find_sents(self, text, exact=False):
        text = text.lower()
        
        sents = []
        for sent in nltk.sent_tokenize(text):
            if self.find_sent(sent, exact):
                sents.append(sent)

        return sents


class Variations:
    def __init__(self, var_list):
        self.list = list(var_list)
        print('[INFO] There are %d variations in the list' % len(self.list))
        self.var_list = []
        for var in self.list:
            Var = Variation(var)
            self.var_list.append(Var)
    
    def to_df(self):
        var_dicts = [Var.to_dict() for Var in self.var_list]
        df = pd.DataFrame(var_dicts)
        return df

    def to_dummy_df(self):
        dummy_columns = ['start_amino', 'end_amino', 'type']
        df = self.to_df()
        df_dummy = pd.get_dummies(df, columns=dummy_columns)
        return df_dummy

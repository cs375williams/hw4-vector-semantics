"""
CS124 PA5: Quizlet // Stanford, Winter 2020
by @lcruzalb, with assistance from @jchen437
"""
import csv
import sys
import getopt
import os
import math
import operator
import random
from collections import defaultdict
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
import spacy

#############################################################################
###                    CS124 Homework 5: Quizlet!                         ###
#############################################################################

# ------------------------- Do not modify code below --------------------------------
        
class Part2B_Runner():
    def __init__(self, find_synonym):        
        self.find_synonym = find_synonym

        # load embeddings
        self.embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)

        # load questions
        root_dir = 'data/dev/'
        self.synonym_qs = load_synonym_qs(root_dir + 'synonyms.csv')

    def evaluate(self, print_q=True):
        print ('Part 1: Synonyms!')
        print ('-----------------')

        acc_euc_dist = self.get_synonym_acc('euc_dist', self.embeddings, self.synonym_qs, print_q)
        acc_cosine_sim = self.get_synonym_acc('cosine_sim', self.embeddings, self.synonym_qs, print_q)

        print ('accuracy using euclidean distance: %.5f' % acc_euc_dist)
        print ('accuracy using cosine similarity : %.5f' % acc_cosine_sim)
        

        print (' ')
        return acc_euc_dist, acc_cosine_sim

    def get_synonym_acc(self, comparison_metric, embeddings, synonym_qs, print_q=False):
        '''
        Helper function to compute synonym answering accuracy
        '''
        if print_q:
            metric_str = 'cosine similarity' if comparison_metric == 'cosine_sim' else 'euclidean distance'
            print ('Answering part 1 using %s as the comparison metric...' % metric_str)

        n_correct = 0
        for i, (w, choices, answer) in enumerate(synonym_qs):
            ans = self.find_synonym(w, choices, embeddings, comparison_metric)

            if ans == answer: n_correct += 1

            if print_q:
                print ('%d. What is a synonym for %s?' % (i+1, w))
                a, b, c, d = choices[0], choices[1], choices[2], choices[3]
                print ('    a) %s\n    b) %s\n    c) %s\n    d) %s' % (a, b, c, d))
                print ('you answered: %s \n' % ans)

        acc = n_correct / len(synonym_qs)
        return acc

class Part2C_Runner():
    def __init__(self, find_analogy_word):        
        self.find_analogy_word = find_analogy_word

        # load embeddings
        self.embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)

        # load questions
        root_dir = 'data/dev/'
        self.analogy_qs = load_analogy_qs(root_dir + 'analogies.csv')

    def evaluate(self, print_q=False):
        '''
        Calculates accuracy on part 2.
        '''
        print ('Part 2: Analogies!')
        print ('------------------')

        n_correct = 0
        for i, (tup, choices) in enumerate(self.analogy_qs):
            a, b, aa, true_bb = tup
            ans = self.find_analogy_word(a, b, aa, choices, self.embeddings)
            if ans == true_bb: n_correct += 1

            if print_q:
                print ('%d. %s is to %s as %s is to ___?' % (i+1, a, b, aa))
                print ('    a) %s\n    b) %s\n    c) %s\n    d) %s' % tuple(choices))
                print ('You answered: %s\n' % ans)

        acc = n_correct / len(self.analogy_qs)
        print ('accuracy: %.5f' % acc)
        print (' ')
        return acc

class Part2D_Runner():
    def __init__(self, get_similarity):        
        self.get_similarity = get_similarity

        # load embeddings
        self.embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)

        # load questions
        root_dir = 'data/dev/'
        self.sentence_sim_qs = load_sentence_sim_qs(root_dir + 'sentences.csv')

    def evaluate(self, print_q=False):
        '''
        Calculates accuracy of part 3.
        '''
        print ('Part 3: Sentence similarity!')
        print ('----------------------------')

        acc_base = self.get_sentence_sim_accuracy(self.embeddings, self.sentence_sim_qs, print_q=print_q)
        print ('accuracy (regular): %.5f' % acc_base)
        acc_POS = None 

        # if self.score_pos: 
        #     acc_POS = self.get_sentence_sim_accuracy(self.embeddings, self.sentence_sim_qs, use_POS=True, print_q=print_q)
        #     print ('accuracy with POS weighting: %.5f' % acc_POS)
        # print (' ')
        return acc_base

    #def get_sentence_sim_accuracy(self, embeddings, sentence_sim_qs, use_POS, print_q=False):
    def get_sentence_sim_accuracy(self, embeddings, sentence_sim_qs, print_q=False):
        '''
        Helper function to compute sentence similarity classification accuracy.
        '''
        THRESHOLD = 0.95
        #POS_weights = self.load_pos_weights_map() if use_POS else None

        if print_q:
            #type_str = 'with POS weighting' if use_POS else 'regular'
            #print ('Answering part 3 (%s)...' % type_str)
            print ('Answering part 3...')

        n_correct = 0
        for i, (label, s1, s2) in enumerate(sentence_sim_qs):
            sim = self.get_similarity(s1, s2, embeddings)
            pred = 1 if sim > THRESHOLD else 0
            if pred == label: n_correct += 1

            if print_q:
                print ('%d. True/False: the following two sentences are semantically similar:' % (i+1))
                print ('     1. %s' % s1)
                print ('     2. %s' % s2)
                print ('You answered: %r\n' % (True if pred == 1 else False))

        acc = n_correct / len(sentence_sim_qs)
        return acc

class Part2E_Runner():
    def __init__(self, occupation_exploration):        
        self.occupation_exploration = occupation_exploration

        # load embeddings
        self.embeddings = KeyedVectors.load_word2vec_format("data/embeddings/glove50_4k.txt", binary=False)


    def evaluate(self):
        '''
        Runs part 4 functions
        '''
        print ('Part 4: Exploration!')
        print ('--------------------')

        occupations = load_occupations_list()
        #top_man_occs, top_woman_occs = self.occupation_exploration(occupations, self.embeddings)
        out = self.occupation_exploration(occupations, self.embeddings)
        top_man_occs = out['man']
        top_woman_occs = out['woman']

        print ('occupations closest to "man" - you answered:')
        for i, (occ, sim) in enumerate(top_man_occs):
            print(f"\t {i}. {occ} (cosine={sim})")
        print ('occupations closest to "woman" - you answered:')
        for i, (occ, sim) in enumerate(top_woman_occs):
            print(f"\t {i}. {occ} (cosine={sim})")

        # # sanity check they answered written - this is just a heuristic
        # written_ans = self.part4_written()
        # if 'TODO' in written_ans:
        #     print ('Part 4 written answer contains TODO, did you answer it?')
        # print (' ')
        return top_man_occs, top_woman_occs

def load_synonym_qs(filename):
    '''
    input line:
        word    c1,c2,c3,c4     answer

    returns list of tuples, each of the form:
        (word, [c1, c2, c3, c4], answer)
    '''
    synonym_qs = []
    with open(filename) as f:
        f.readline()    # skip header
        for line in f:
            word, choices_str, ans = line.strip().split('\t')
            choices = [c.strip() for c in choices_str.split(',')]
            synonym_qs.append((word.strip(), choices, ans.strip()))
    return synonym_qs

def load_analogy_qs(filename):
    '''
    input line:
        a,b,aa,bb   c1,c2,c3,c4

    returns list of tuples, each of the form:
        (a, b, aa, bb)  // for analogy a:b --> aa:bb
    '''
    analogy_qs = []
    with open(filename) as f:
        f.readline()    # skip header
        for line in f:
            toks, choices_str = line.strip().split('\t')
            analogy_words = tuple(toks.strip().split(','))          # (a, b, aa, bb)
            choices = [c.strip() for c in choices_str.split(',')]   # [c1, c2, c3, c4]
            analogy_qs.append((analogy_words, choices))
    return analogy_qs

def load_sentence_sim_qs(filename):
    '''
    input line:
        label   s1  s2
    
    returns list of tuples, each of the form:
        (label, s1, s2)
    '''
    samples = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            label_str, s1, s2 = line.split('\t')
            label = int(label_str)
            samples.append((label, s1.strip(), s2.strip()))
    return samples

def load_occupations_list():
    '''
    Helper that loads the list of occupations for part 4
    '''
    occupations = []
    with open("data/occupations.txt") as f:
        for line in f:
            occupations.append(line.strip())
    return occupations

def main():
    print("Run homework assignment in pa5-quizlet.ipynb")

if __name__ == "__main__":
        main()

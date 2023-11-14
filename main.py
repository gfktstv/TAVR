import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

import re

from itertools import chain

with open('wt2.txt', 'r') as file:
    text1 = file.read().replace('\n', '')


class Text:
    """
    Class to work with a whole text of an essay

    Arguments:
        source (str): Source of Data consisting a text and a question

    Attributes:
        content (str): The content of an essay

    """

    def __init__(self, source):
        self.source = source
        with open(f'{self.source}', 'r') as file:
            self.content = file.read().replace('\n', ' ')


class Paragraph:
    """
    Class to work with a paragraph from the text
    """


class Sentence:
    """
    Class to work with a sentence
    """


class Word:
    """
    Class to get characteristics of a word

    Description:
        Class that shows several characteristics of a word
        such as list of synonyms, list of lemmas,
        count of repetition in the text and so on. All these
        parameters are need to evaluate Lexical Resource of the text.

    Arguments:
        word_couple (list): word and part of speech

    Attributes:
        content (str): word itself
        list_of_meanings (list): synset from wordnet
        list_of_all_synonyms (list): all synonyms of the word

    Methods:

    """

    def __init__(self, word_couple):
        self.word_couple = word_couple
        self.content = self.word_couple[0]
        self.list_of_meanings = wn.synsets(f'{self.word_couple[0]}', pos=self.word_couple[1])

        self.list_of_all_synonyms = []
        for synset in self.list_of_meanings:
            self.list_of_all_synonyms.append(
                [str(lemma.name()) for lemma in synset.lemmas()]
            )


class LexicalResourceStats:
    """
    Abstract class to evaluate lexical resource criteria

    Arguments:
        text_class_variable (Text): text of an essay

    Attributes:
        content (str): The content of an essay
        list_of_all_words (list): List of all words
    """
    pass

    def __init__(self, text_class_variable):
        self.content = text_class_variable.content
        self.list_of_all_words = nltk.word_tokenize(self.content.lower())
        self.list_of_all_words = nltk.pos_tag(self.list_of_all_words)

        # delete punctuations marks
        punctuation_marks_list = [
            '!', '.', ',', ';', ':'
                                '...', ')', '(', '"', '',
            "'", '-', '--', '?', '—',
            '[', ']'
        ]

        for word_tuple in self.list_of_all_words:
            if word_tuple[0] in punctuation_marks_list:
                self.list_of_all_words.remove(word_tuple)


class SemanticRelationsStats(LexicalResourceStats):
    """
    Child class of LexicalResourceStats to evaluate semantic relations

    Attributes:
        intermediate_list_of_words (list): auxiliary attribute
        words_for_semantic_connections_evaluating (list): list of words suitable for evaluating

    Methods:
        semantic_relations_algorithm_1: first algorithm to evaluate semantic relations
    """

    def __init__(self, text_class_variable):
        super().__init__(text_class_variable)
        #
        # create a list of words that suitable for semantic relations evaluating
        #
        self.intermediate_list_of_words = self.list_of_all_words
        self.intermediate_list_of_words = list(set(self.intermediate_list_of_words))

        # delete stopwords
        for word_couple in self.intermediate_list_of_words:
            if word_couple[0] in stopwords.words('english'):
                self.intermediate_list_of_words.remove(word_couple)

        # delete unnecessary parts of speech
        necessary_pos_list = [
            'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'VBZ',
            'VBP', 'VBN', 'VBD', 'VBG', 'VB', 'RBS', 'RB', 'RBR'
        ]
        self.words_for_semantic_connections_evaluating = []
        for word_couple in self.intermediate_list_of_words:
            if word_couple[1] in necessary_pos_list:
                self.words_for_semantic_connections_evaluating.append(word_couple)

        # convert word_tuple into word_list (word_couple)
        for i in range(len(self.words_for_semantic_connections_evaluating)):
            self.words_for_semantic_connections_evaluating[i] = list(self.words_for_semantic_connections_evaluating[i])

        # convert nltk pos_tags into wordnet pos_tags
        noun_pattern = re.compile(
            '^N.*'
        )

        verb_pattern = re.compile(
            '^V.*'
        )

        adjective_pattern = re.compile(
            '^J.*'
        )

        adverb_pattern = re.compile(
            '^R.*'
        )

        for word_couple in self.words_for_semantic_connections_evaluating:
            word_couple[1] = noun_pattern.sub(wn.NOUN, word_couple[1])
            word_couple[1] = verb_pattern.sub(wn.VERB, word_couple[1])
            word_couple[1] = adjective_pattern.sub(wn.ADJ, word_couple[1])
            word_couple[1] = adverb_pattern.sub(wn.ADV, word_couple[1])

        # create Word class objects
        for i in range(len(self.words_for_semantic_connections_evaluating)):
            self.words_for_semantic_connections_evaluating[i] = Word(self.words_for_semantic_connections_evaluating[i])

    def semantic_relations_algorithm_1(self):
        """
        Method to evaluate semantic relations by first algorithm

        Description:
            Evaluating consists of a few steps:
            1. Sort a list of words by a number of synonyms (synonyms for each value of word)
            2. Starting from a first element of the list (this element will have maximum synonyms)
               we check each next word if that word in the list of synonyms
            2.1. If next word is in list of synonyms than we calculate this as 1 semantic relation
            2.2. If next word is not in list of synonyms -- we do nothing
            3. When we check each word we move to the second word and start cycle again
        :return:
        """
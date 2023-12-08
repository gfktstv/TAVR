import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer

import spacy
from spacy import displacy
import en_core_web_lg

from enchant.checker import SpellChecker
import enchant

import re

import itertools
from itertools import chain

with open('wt2.txt', 'r') as file:
    text1 = file.read().replace('\n', '')

nlp = en_core_web_lg.load()


class Text:
    """
    Class to work with a whole text of the essay

    Arguments:
        source (str): Source of Data consisting an essay

    Attributes:
        content (str): Content of the essay
        nlp_doc (spaCy doc): Doc object which contains tokens from the essay

    """

    def __init__(self, source):
        self.source = source
        with open(f'{self.source}', 'r') as file:
            self.content = file.read().replace('\n', ' ')

        self.nlp_doc = nlp(self.content)

    def __str__(self):
        return text.content


# possible that I will not use class Word because I have spaCy tokens with all features I need
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
        content_stemmed (str): word stem (without affixes)
        part_of_speech (str): part of speech of word
        list_of_meanings (list): synset from wordnet
        list_of_all_synonyms (list): all synonyms of the word

    Methods:

    """

    def __init__(self, word_couple):
        self.word_couple = word_couple
        self.content = self.word_couple[0]
        self.content_stemmed = SnowballStemmer('english').stem(self.content)

        self.part_of_speech = self.word_couple[1]

        self.list_of_meanings = wn.synsets(f'{self.word_couple[0]}', pos=self.word_couple[1])

        self.list_of_all_synonyms = []
        for synset in self.list_of_meanings:
            self.list_of_all_synonyms.append(
                [str(lemma.name()) for lemma in synset.lemmas()]
            )
        self.list_of_all_synonyms = list(chain.from_iterable(self.list_of_all_synonyms))

        self.count_of_occurrences = 0

    def get_count_of_occurrences(self, list_of_stems):
        self.count_of_occurrences = list_of_stems.count(self.content_stemmed)


class LexicalResourceStats:
    """
    Abstract class to evaluate lexical resource criteria

    Arguments:
        text_class_variable (Text): Text class variable containing the content and the doc object

    Attributes:
        content (str): Content of the essay
        nlp_doc (spaCy doc): Doc object which contains tokens from the essay
        words_without_stopwords_and_punct_marks (list): List of words without stopwords and punctuation marks
    """

    def __init__(self, text_class_variable):
        self.content = text_class_variable.content
        self.nlp_doc = text_class_variable.nlp_doc

        self.words_without_stopwords_and_punct_marks = []
        for token in self.nlp_doc:
            if not token.is_stop and not token.is_punct:
                self.words_without_stopwords_and_punct_marks.append(token)


class WordSpellingAndFormationStats(LexicalResourceStats):
    """
    Child class of LexicalResourceStats to check word spelling and formation mistakes

    Attributes:
    Methods:
    """

    def __init__(self, text_class_variable):
        super().__init__(text_class_variable)
        self.spelling_checker = SpellChecker('en')
        self.spelling_checker.set_text(self.content)
        self.errors_counter = 0

    def get_word_spelling_and_formation_stats(self):
        for err in self.spelling_checker:
            self.errors_counter += 1

        return self.errors_counter


class SemanticRelationsStats(LexicalResourceStats):
    """
    Child class of LexicalResourceStats to evaluate semantic relations

    Attributes:
        semantic_field (set): Set of words that semantically related to the question
        unique_tokens (set): Unique token from the content

        pos_equal_words_pairs (list): Pairs of words which lemmas are not equal and pos are equal
        semantic_related_pairs (list): Pairs of words that have semantic connection
        semantic_chains (list): Chain of semantic related words

    Methods:
        get_statistics_of_semantic_relations
    """

    def __init__(self, text_class_variable):
        super().__init__(text_class_variable)
        self.semantic_field = set()

        self.unique_tokens = set()
        for token in self.words_without_stopwords_and_punct_marks:
            self.unique_tokens.add(token)

        # for get_statistics_of_semantic_relations
        self.pos_equal_words_pairs = list()
        self.semantic_related_pairs = list()
        self.semantic_chains = list()

    def get_statistics_of_semantic_relations(self):
        """
                Method to evaluate semantic relations by first algorithm

                Description of algorithm:
                    1. Create a set of pairs of all unique words
                    2. We take each pair and check 2 parameters: similarity (nltk) and amount of values
                        2.1 We consider pairs of words as semantic relational when similarity is more than
                        2.2 We assign a weight (.25, .5, 1) of semantic relation by amount of values of words in pair
                :return:
        """

        # create a list with pairs of words which lemmas are not equal and pos are equal
        self.pos_equal_words_pairs = list(
            [x, y] for x in self.unique_tokens for y in self.unique_tokens if x.lemma_ != y.lemma_ if x.pos_ == y.pos_
        )

        # create a list with semantic related pairs of words
        for pair in self.pos_equal_words_pairs:
            if pair[0].similarity(pair[1]) > 0.7:
                self.semantic_related_pairs.append(pair)

        # create a list with semantic chains
        # semantic chain is all semantically related words from the content
        for i in range(len(self.semantic_related_pairs)):
            semantic_chain = [self.semantic_related_pairs[i]]
            for j in range(i + 1, len(self.semantic_related_pairs)):
                if len(list(set(self.semantic_related_pairs[i]).intersection(set(self.semantic_related_pairs[j])))) > 0:
                    semantic_chain.append(self.semantic_related_pairs[j])
            self.semantic_chains.append(
                list(set(chain.from_iterable(semantic_chain)))
            )

        print(self.semantic_chains)

        self.semantic_chains = list()
        semantic_chain = set()
        self.unique_tokens = list(self.unique_tokens)
        for i in range(0, 5):
            semantic_chain = set(
                x.lemma_ for x in self.unique_tokens if self.unique_tokens[i].similarity(x) > 0.7
            )
            semantic_chain.add(self.unique_tokens[i].lemma_)
            self.semantic_chains.append(semantic_chain)
            self.unique_tokens = list(
                set(self.unique_tokens) - semantic_chain
            )

        return self.semantic_chains

import spacy
from spacy.tokens import Doc
from spacy_ngram import NgramComponent

from nltk.corpus import wordnet as wn

from lexical_diversity import lex_div as ld
from corpus_toolkit import corpus_tools as ct

from statistics import mean

import pandas as pd
import numpy as np

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacy-ngram')  # Pipeline for n-gram marking


class Text:
    """
    Splits text into tokens and n-grams, deletes stop words and does other primary text processing.

    Arguments:
        source (str): Source of Data consisting an essay

    Attributes:
        content (str): Content of the essay
        nlp_doc (spaCy doc): Doc object which contains tokens from the essay

    Methods:
        get_tokens
        get_lemmas
        get_types
        get_bigrams
        get_trigrams

    """

    def __init__(self, source):
        # self.source = source
        # with open(f'{self.source}', 'r') as file:
        #     self.content = file.read().replace('\n', ' ')
        self.content = source

        self.nlp_doc = nlp(self.content)

    def __str__(self):
        return self.content

    def get_tokens(self, punct=False, insignificant=True):
        """
        Returns a list of tokens with or without punctuation marks and insignificant for lexical measures words.
        Insignificant words are stopwords or words with one of these part of speech: proper noun, symbol, particle,
        coordinating conjunction, adposition or unknown part of speech
        """
        if punct:
            tokens = [token for token in self.nlp_doc]
        else:
            tokens = [token for token in self.nlp_doc if not token.is_punct]

        banned_pos = ['PROPN', 'SYM', 'PART', 'CCONJ', 'ADP', 'X']
        if not insignificant:
            tokens = [token for token in tokens if (not token.is_stop) and (token.pos_ not in banned_pos)]

        return tokens

    def get_lemmas(self, insignificant=True):
        """Returns a list of lemmas with or without insignificant for lexical measures words (see get_tokens)"""
        if insignificant:
            return [token.lemma_ for token in self.get_tokens(False, True)]
        else:
            return [token.lemma_ for token in self.get_tokens(False, False)]

    def get_types(self, insignificant=True):
        """
        Returns a list of types with or without insignificant for lexical measures words (see get_tokens)
        """
        types = list()
        unique_tokens_text = list()
        if insignificant:
            for token in self.get_tokens():
                if token.text not in unique_tokens_text:
                    types.append(token)
                    unique_tokens_text.append(token.text)
            return types
        else:
            unique_tokens_text = list()
            for token in self.get_tokens(insignificant=False):
                if token.text not in unique_tokens_text:
                    types.append(token)
                    unique_tokens_text.append(token.text)
            return types

    def get_bigrams(self):
        """Returns a list of bigrams. Tokens in bigrams divided by __"""
        bigrams = list()
        for i in range(len(self.get_tokens()) - 2):
            bigrams.append(f'{self.get_tokens()[i]}__{self.get_tokens()[i + 1]}'.lower())
            bigrams.append(f'{self.get_tokens()[i]}__{self.get_tokens()[i + 2]}'.lower())
        return bigrams

    def get_trigrams(self):
        """Returns a list of trigrams. Tokens in trigrams divided by __"""
        trigrams = list()
        for i in range(len(self.get_tokens()) - 2):
            trigrams.append(f'{self.get_tokens()[i]}__{self.get_tokens()[i + 1]}__{self.get_tokens()[i + 2]}'.lower())
        return trigrams


class LexicalSophisticationMeasurement:
    """
    Class measures lexical sophistication of a given text. It contains several methods such as
    word frequency, N-gram statistics, word range, word specificity, vocabulary CEFR level
    and academic vocabulary content. All of them return measurements organized by dictionaries.
    Also class marks up words and n-grams with frequency, range and other characteristics.

    Arguments:
        text_class_variable: variable belonging to the class Text

    Attributes:
        content (str): Content of the essay
        tokens (list): All tokens from the text (without punctuation's marks)
        significant_tokens (list): Tokens without stop words
        bigrams (list): 2-grams based on tokens
        trigrams (list): 3-grams based on tokens

        marked_up_tokens (dict of dict): Dictionary with key of a token and value of a token_dict which
        represents characteristics of a token (stopword, frequency, range, academic, level CEFR)
        marked_up_n_grams (dict of dict): Dictionary with key of an n-gram and value of an n_gram_dict which
        represents frequency and range of an n-gram

    Methods:
        word_freq_range
        n_gram_freq_range
        academic_formulas_freq
        academic_vocabulary_content
        vocabulary_by_level
        word_specificity

    """

    def __init__(self, text_class_variable=None, token_list=None, pos_tag=None):
        """
        The input can be both Text's instance or a list of tokens. The last one is used for replacement options,
        and it takes as input list with tokens and their part of speech tag (because they are synonyms they have
        similar pos tags)
        """
        if text_class_variable is not None:
            assert isinstance(text_class_variable, Text)
            self.content = text_class_variable.content
            self.tokens = text_class_variable.get_tokens()
            self.significant_tokens = text_class_variable.get_tokens(False, False)
            self.bigrams = text_class_variable.get_bigrams()
            self.trigrams = text_class_variable.get_trigrams()
        # For replacement options
        elif token_list is not None:
            assert isinstance(token_list, list)
            doc = Doc(nlp.vocab, words=token_list, pos=[pos_tag for i in range(len(token_list))], lemmas=token_list)
            banned_pos = ['PROPN', 'SYM', 'PART', 'CCONJ', 'ADP', 'X']
            self.tokens = [token for token in doc if (not token.is_punct)]
            self.significant_tokens = [token for token in doc if (not token.is_stop)
                                       and (token.pos_ not in banned_pos) and (not token.is_punct)]

        # Dictionary consisting of token and token_dict.
        # Token dict is a dictionary with frequency, range, academic and level keys
        self.marked_up_tokens = dict()
        for token in self.tokens:
            if not token.is_stop:
                self.marked_up_tokens[token] = {
                    'stopword': False, 'freq': 0, 'range': 0,
                    'academic': bool(), 'level': None
                }
            if token.is_stop:
                self.marked_up_tokens[token] = {'stopword': True}

        # Dictionary consisting of n-gram and n-gram_dict.
        # N-gram dict is a dictionary with frequency and range.
        # Dictionary includes academic formulas marked with frequency and occurrences in a text as well
        self.marked_up_n_grams = dict()

    def word_freq_range(self, for_replacement_options=False):
        """
        Add frequency and range of a token to self.marked_up_tokens.
        Calculates average word frequency & range.

        Returns: dict containing average frequency and range.

        """
        # Load and read tagged corpus, then tokenize (and lemmatize) it and create a frequency and range dictionary
        brown_freq = ct.frequency(ct.tokenize(ct.ldcorpus('corpora/tagged_brown', verbose=False)))
        brown_range = ct.frequency(ct.tokenize(ct.ldcorpus('corpora/tagged_brown', verbose=False)), calc='range')

        # Create lists of frequencies and ranges for calculating average values further
        word_frequencies, word_ranges = list(), list()
        for token in self.significant_tokens:
            try:
                word = f'{token.lemma_}_{token.pos_}'.lower()
                word_freq = brown_freq[word]
                word_range = brown_range[word]
                self.marked_up_tokens[token]['freq'] = word_freq
                self.marked_up_tokens[token]['range'] = word_range

                word_frequencies.append(word_freq)
                word_ranges.append(word_range)
            except KeyError:
                # Ignores KeyError that occurs due to the absence of a token in the corpus
                continue

        if not for_replacement_options:
            measurements_dict = {
                'Word frequency average': mean(word_frequencies),
                'Word range average': mean(word_ranges)
            }

            return measurements_dict

    def n_gram_freq_range(self):
        """
        Create lists with bi-, trigrams and its frequency & range.
        Calculates average bi-, trigrams frequency & range.

        Returns:
            1) dict containing average frequency and range,
            2) list of bigrams marked with frequency and range,
            3) list of trigrams marked with frequency and range.


        """
        # Create n-grams frequency and range dictionary
        brown_bigram_freq = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=2)
        )
        brown_bigram_range = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=2), calc='range'
        )
        brown_trigram_freq = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=3)
        )
        brown_trigram_range = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=3), calc='range'
        )

        # Create lists of frequencies and ranges for calculating average values further
        bigram_frequencies, bigram_ranges = list(), list()
        trigram_frequencies, trigram_ranges = list(), list()
        for bigram in self.bigrams:
            try:
                bigram_freq = brown_bigram_freq[bigram]
                bigram_range = brown_bigram_range[bigram]

                bigram_frequencies.append(bigram_freq)
                bigram_ranges.append(bigram_range)

                # Add bigram marked with frequency and range to a dict
                self.marked_up_n_grams[bigram] = {'freq': bigram_freq, 'range': bigram_range}
            except KeyError:
                continue
        for trigram in self.trigrams:
            try:
                trigram_freq = brown_trigram_freq[trigram]
                trigram_range = brown_trigram_range[trigram]

                trigram_frequencies.append(trigram_freq)
                trigram_ranges.append(trigram_range)

                # Add trigram marked with frequency and range to a dict
                self.marked_up_n_grams[trigram] = {'freq': trigram_freq, 'range': trigram_range}
            except KeyError:
                continue

        measurements_dict = {
            'Bigram frequency average': mean(bigram_frequencies),
            'Bigram range average': mean(bigram_ranges),
            'Trigram frequency average': mean(trigram_frequencies),
            'Trigram range average': mean(trigram_ranges)
        }

        return measurements_dict

    def academic_formulas_freq(self):
        """
        Create list with academic formulas and their frequency and amount of occurrences.
        Calculates average frequency.

        Returns:
            1) list of academic frequencies marked with frequency,
            2) dict containing average frequency.

        """
        # Converts AFL csv to list
        AFL_csv = pd.read_csv('corpora/AFL.csv')
        academic_formulas_list = [
            [record['Formula'], record['Frequency per million']] for record in AFL_csv.to_dict('records')
        ]

        frequencies = 0
        occurrences = 0
        for formula in academic_formulas_list:
            if formula[0] in self.content:
                frequencies += int(formula[1]) * self.content.count(formula[0])
                occurrences += self.content.count(formula[0])

                self.marked_up_n_grams[formula[0]] = {
                    'freq': int(formula[1]), 'range': None, 'occur': self.content.count(formula[0])
                }

        if occurrences == 0:
            measurements_dict = {
                'Academic formulas frequency': 0
            }
        else:
            measurements_dict = {
                'Academic formulas frequency': frequencies / occurrences
            }

        return measurements_dict

    def academic_vocabulary_content(self):
        """
        Calculates amount of academic words (words from Academic Word List) and the ratio
        between the number of academic words and the number of tokens (not counting stopwords).

        Returns: dictionary with amount of academic words and their percentage of the text.

        """
        academic_word_list = list()
        with open('corpora/AWL.txt', 'r') as file:
            for row in file:
                academic_word_list.append(row.rstrip('\n'))

        academic_words = list()
        for token in self.significant_tokens:
            if token.text in academic_word_list:
                self.marked_up_tokens[token]['academic'] = True
                academic_words.append(token)

        statistics_dict = {
            'Amount of academic words': len(academic_words),
            'Percentage of academic words': len(academic_words) / len(self.significant_tokens)
        }

        return statistics_dict

    def vocabulary_by_level(self, for_replacement_options=False):
        """
        Calculates amount of tokens for each CEFR level using 'words by level' dir and
        percentage of words for each category (further visualized by a pie plot).

        Returns: dictionary with amount of words for each CEFR level, their percentage of the text.

        """
        levels = {
            2: 'A1', 3: 'A2', 4: 'B1', 5: 'B2', 6: 'C1'
        }
        efllex_corpus = pd.read_csv('corpora/EFLLex_NLP4J', delimiter='\\t', engine='python')
        tokens_with_CEFR_level_corpus = dict()
        for i in range(len(efllex_corpus.iloc[:, 0])):
            for level in [2, 3, 4, 5, 6]:
                level_frequency = efllex_corpus.iloc[i, level]
                if level_frequency > 0:  # Selects first level where the token appears
                    # The key in dictionary is tuple with token lemma, and it's part of speech tag,
                    # the value is its level
                    lemma_and_pos = (efllex_corpus.iloc[i, 0], efllex_corpus.iloc[i, 1])
                    tokens_with_CEFR_level_corpus[lemma_and_pos] = levels[level]
                    break

        tokens_by_CEFR_level = {
            'A1': list(), 'A2': list(),
            'B1': list(), 'B2': list(),
            'C1': list(), 'C2': list()
        }
        # Dict to convert spaCy's tags to the corpus' tags. Some of the tags are empty (None) because they are
        # insignificant or missing in the corpus's tags
        tags_dict = {
            'ADJ': 'JJ', 'ADP': None,
            'ADV': 'RB', 'AUX': 'VB',
            'CCONJ': None, 'DET': None,
            'INTJ': None, 'NOUN': 'NN',
            'NUM': 'CD', 'PART': None,
            'PRON': [' NN', 'EX', 'PRP', 'WP'], 'SCONJ': 'IN',
            'VERB': ['MD', 'VB'], 'X': 'XX',
            'PROPN': None, 'PUNCT': None,
            'SPACE': None, 'SYM': None,
        }
        for token in self.significant_tokens:
            if type(tags_dict[token.pos_]) is str:
                # Some words may not be in corpus, therefore we will use try/except
                try:
                    level = tokens_with_CEFR_level_corpus[(token.lemma_, tags_dict[token.pos_])]
                    tokens_by_CEFR_level[level].append(token)
                    self.marked_up_tokens[token]['level'] = level
                except KeyError:
                    self.marked_up_tokens[token]['level'] = 'C2'
                    tokens_by_CEFR_level['C2'].append(token)
            # If there are several tags for one spaCy tag we try all of them
            elif type(tags_dict[token.pos_]) is list:
                for tag_option in tags_dict[token.pos_]:
                    try:
                        level = tokens_with_CEFR_level_corpus[(token.lemma_, tag_option)]
                        tokens_by_CEFR_level[level].append(token)
                        self.marked_up_tokens[token]['level'] = level
                    except KeyError:
                        pass
                if self.marked_up_tokens[token]['level'] is None:
                    self.marked_up_tokens[token]['level'] = 'C2'
                    tokens_by_CEFR_level['C2'].append(token)
            elif tags_dict[token.pos_] is None:
                continue
            else:
                print(f'Token {token.text} has unknown part of speech tag that is {token.pos_}')

        vocabulary_by_level_dict = {
            'A1 words': len(tokens_by_CEFR_level['A1']), 'A2 words': len(tokens_by_CEFR_level['A2']),
            'B1 words': len(tokens_by_CEFR_level['B1']), 'B2 words': len(tokens_by_CEFR_level['B2']),
            'C1 words': len(tokens_by_CEFR_level['C1']), 'C2 words': len(tokens_by_CEFR_level['C2']),
        }

        level_weight = {
            'A1': 1, 'A2': 1, 'B1': 2, 'B2': 4, 'C1': 8, 'C2': 16
        }

        vocabulary_metric = list()
        for level in tokens_by_CEFR_level.keys():
            for i in range(vocabulary_by_level_dict[f'{level} words']):
                vocabulary_metric.append(i * level_weight[level])

        if not for_replacement_options:
            vocabulary_metric_dict = {
                'Vocabulary': np.average(vocabulary_metric)
            }

            return vocabulary_metric_dict

    def get_full_data(self):
        """
        Combines data from all methods except into one dictionary.

        Returns: dictionary with data from all methods
        """
        word_freq_range_data = self.word_freq_range()
        n_gram_freq_range_data = self.n_gram_freq_range()
        academic_vocabulary_data = self.academic_vocabulary_content()
        academic_formulas_data = self.academic_formulas_freq()
        vocabulary_by_level_data = self.vocabulary_by_level()

        full_data = word_freq_range_data | n_gram_freq_range_data | academic_vocabulary_data \
                    | academic_formulas_data | vocabulary_by_level_data

        return full_data


class LexicalDiversityMeasurement:
    """
    Class measures lexical diversity using TTR, MTLD and MTLD MA Wrap (MTLD-W) indexes.
    Returns a dictionary with measurements.

    Attributes:
        lemmatized_text (list): lemmas from a content except insignificant ones

    Methods:
        indexes_data

    """

    def __init__(self, text_class_variable):
        self.lemmatized_text = text_class_variable.get_lemmas(insignificant=False)

    def indexes_data(self):
        data_dict = {
            'TTR': ld.ttr(self.lemmatized_text),
            'MTLD': ld.mtld(self.lemmatized_text),
            'MTLD MA Wrap': ld.mtld_ma_wrap(self.lemmatized_text)
        }
        return data_dict


class LevelAndDescription:
    """
    Collects data from methods of Measurements’ classes and organizes it in CSV table.
    Evaluates text using this table by CEFR level

    Arguments:
        text_class_variable: variable belonging to the class Text

    Attributes:

    Methods:

    """

    def __init__(self, text_class_variable):
        self.lexical_diversity_data = LexicalDiversityMeasurement(text_class_variable).indexes_data()
        self.lexical_sophistication_data = LexicalSophisticationMeasurement(text_class_variable).get_full_data()

        self.description_dict = self.lexical_diversity_data | self.lexical_sophistication_data


class TextMarkup:
    """

    """

    def __init__(self):
        pass


class TokenReplacementOptions:
    """
    Selects synonyms with lower frequency or lower range or higher CEFR level to a given token.
    It should be mentioned that suggested synonyms might be inappropriate in a text because of different semantics
    that does not count.

    Attributes:
        marked_up_tokens(dict of dicts): tokens from a text marked up with level, frequency and range

    Methods:
        spacy_pos_to_wordnet_pos: Transforms spaCy part of speech tag to wordnet pos tag
        get_synonyms: Returns synonyms of a token based on part of speech marked with level, frequency and range
        get_replacement_options: Returns replacement options based on vocabulary level, frequency and range
    """
    def __init__(self, marked_up_tokens):
        # Assigns dictionary of tokens from a text marked up with level, frequency and range
        self.marked_up_tokens = marked_up_tokens

    @staticmethod
    def spacy_pos_to_wordnet_pos(spacy_pos):
        """Transforms spaCy part of speech tag to wordnet pos tag"""
        if spacy_pos.startswith('N'):
            return wn.NOUN
        elif spacy_pos.startswith('V'):
            return wn.VERB
        elif spacy_pos.startswith('J'):
            return wn.ADJ
        elif spacy_pos.startswith('R'):
            return wn.ADV
        else:
            return None

    def get_synonyms(self, token):
        """Returns synonyms of a token based on part of speech"""
        pos_tag = self.spacy_pos_to_wordnet_pos(token.pos_)
        synonyms = list()
        for synset in wn.synsets(token.text, pos=pos_tag):
            # synonym = synset.lemmas()[0].name()
            # if synonym not in synonyms:
            #     synonyms.append(synonym)
            for synonym in synset.lemmas():
                if synonym.name() not in synonyms:
                    synonyms.append(synonym.name())
        return synonyms

    def get_replacement_options(self, token):
        """
        Returns replacement options based on synonyms of a token excluding synonyms with lower CEFR level,
        higher frequency & range and ones which are not in EFLLex corpus.
        """
        # Assigns instance of LexicalSophisticationMeasurements for synonyms
        lexical_sophistication = LexicalSophisticationMeasurement(token_list=self.get_synonyms(token),
                                                                  pos_tag=token.pos_)
        # Marks up synonymic tokens with frequency and range
        lexical_sophistication.word_freq_range(for_replacement_options=True)
        # Marks up synonymic tokens with level
        lexical_sophistication.vocabulary_by_level(for_replacement_options=True)
        marked_up_synonyms = lexical_sophistication.marked_up_tokens

        # For now, we only imagine that we have token level, frequency and range
        token_level = 'B1'  # self.marked_up_tokens[token]['level']
        token_freq = 65  # self.marked_up_tokens[token]['freq']
        token_range = 52  # self.marked_up_tokens[token]['range']
        # Excludes synonyms which are not in EFLLex corpus
        synonyms = {key: value for key, value in marked_up_synonyms.items() if value['level'] != 'C2'}
        # Leaves only synonyms with higher CEFR level or lower frequency or lower range and excludes freq/range equal 0
        levels = ['A1', 'A2', 'B1', 'B2', 'C1']
        synonyms = {key: value for key, value in synonyms.items()
                    if (value['freq'] < token_freq) or (value['range'] < token_range)
                    or (levels.index(value['level']) > levels.index(token_level))
                    and ((value['freq'] != 0) and (value['range'] != 0))}
        # Sorts synonyms by level
        synonyms_sorted_by_level = sorted(synonyms.items(), key=lambda x: x[1]['level'], reverse=True)
        replacements = [synonym_set[0] for synonym_set in synonyms_sorted_by_level]
        return replacements[0:2]


class AnalysisOfVocabularyRichness:
    """

    """

    def __init__(self):
        pass


# Test TokenReplacementOptions
token_test = nlp('expand')[0]
print(TokenReplacementOptions('test').get_replacement_options(token_test))
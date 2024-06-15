import statistics

import spacy
from spacy.tokens import Doc
from spacy_ngram import NgramComponent

from nltk.corpus import wordnet as wn

from lexical_diversity import lex_div as ld
from corpus_toolkit import corpus_tools as ct

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacy-ngram')  # Pipeline for n-gram marking


class _Text:
    """
    Splits text into _tokens and n-grams, deletes stop words and does other preprocessing.

    Arguments:
        essay (str): Essay itself

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

    def __init__(self, essay):
        self.content = essay
        self.nlp_doc = nlp(self.content)

    def __str__(self):
        return self.content

    def get_tokens(self, include_punct=False, include_content_words=True, include_functional_words=True):
        """
        Returns a list of _tokens with or without punctuation marks and insignificant for lexical measures words.

        :param bool include_punct: Include punctuation marks in output list
        :param bool include_content_words: Include content words (those words which are not functional)
        :param bool include_functional_words: Include stopwords, proper nouns, symbols, particles, adposition,
                                           coordinating conjunction and unknown parts of speech
        """
        tokens = list()
        banned_pos = ['PROPN', 'SYM', 'PART', 'CCONJ', 'ADP', 'X']
        if include_content_words and include_functional_words:
            tokens = [token for token in self.nlp_doc]
            # Check for small text or if text not in English
            if len(tokens) < 50:
                raise TokensAreNotRecognized(
                    f'SpaCy module recognized only {len(tokens)} tokens which is less then 50'
                )
        elif include_content_words:
            tokens = [token for token in self.nlp_doc if (not token.is_stop) and (token.pos_ not in banned_pos)]
            # Check for small text or if text not in English
            if len(tokens) < 50:
                raise TokensAreNotRecognized(
                    f'SpaCy module recognized only {len(tokens)} tokens which is less then 50'
                )
        elif include_functional_words:
            tokens = [token for token in self.nlp_doc if token.is_stop or (token.pos_ in banned_pos)]

        if include_punct and (len(tokens) == 0):
            # For a option with only punctuational marks being returned
            tokens = [token for token in self.nlp_doc if token.is_punct]
        elif include_punct is False:
            tokens = [token for token in tokens if not token.is_punct]

        return tokens

    def get_lemmas(self, include_content_words=True, include_functional_words=True):
        """
        Returns a list of lemmas with or without insignificant for lexical measures words.

        :param bool include_content_words: Include content words (those words which are not functional)
        :param bool include_functional_words: Include stopwords, proper nouns, symbols, particles, adposition,
                                           coordinating conjunction and unknown parts of speech
        """
        tokens = list()
        if include_content_words and include_functional_words:
            tokens = self.get_tokens(include_content_words=True, include_functional_words=True)
        elif include_content_words:
            tokens = self.get_tokens(include_content_words=True, include_functional_words=False)
        elif include_functional_words:
            tokens = self.get_tokens(include_content_words=False, include_functional_words=True)

        return [token.lemma_ for token in tokens]

    def get_lemmas_occurrences(self, include_content_words=True, include_functional_words=False):
        """
        Returns a dict of lemmas with number of occurrences

        :param bool include_content_words: Include content words (those words which are not functional)
        :param bool include_functional_words: Include stopwords, proper nouns, symbols, particles, adposition,
                                           coordinating conjunction and unknown parts of speech
        """
        lemmas = list()
        if include_content_words and include_functional_words:
            lemmas = self.get_lemmas(include_content_words=True, include_functional_words=True)
        elif include_content_words:
            lemmas = self.get_lemmas(include_content_words=True, include_functional_words=False)
        elif include_functional_words:
            lemmas = self.get_lemmas(include_content_words=False, include_functional_words=True)

        # Clean from '\n\n' (they are needed in tokens for a web app to preserve paragraphs after marking up an essay)
        lemmas = [lemma for lemma in lemmas if lemma != '\n\n']

        lemmas_occurrences = dict()
        for lemma in lemmas:
            lemmas_occurrences[lemma] = lemmas.count(lemma)

        return lemmas_occurrences

    def get_types(self, include_content_words=True, include_functional_words=True):
        """
        Returns a list of types (words from a text without repetitions).

        :param bool include_content_words: Include content words (those words which are not functional)
        :param bool include_functional_words: Include stopwords, proper nouns, symbols, particles, adposition,
                                           coordinating conjunction and unknown parts of speech
        """
        tokens = list()
        if include_content_words and include_functional_words:
            tokens = self.get_tokens(include_content_words=True, include_functional_words=True)
        elif include_content_words:
            tokens = self.get_tokens(include_content_words=True, include_functional_words=False)
        elif include_functional_words:
            tokens = self.get_tokens(include_content_words=False, include_functional_words=True)

        types = list()
        unique_tokens_text = list()
        for token in tokens:
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


class _LexicalSophisticationMeasurements:
    """
    Class measures lexical sophistication of a given text by average word frequency & range, n-gram frequency & range,
    academic formulas frequency, content of academic vocabulary, level of vocabulary. In addition, it marks up tokens
    and n-grams with frequency, range and other characteristics.

    Note, that all tokens and n-grams will be marked up only after calling all methods or method get_full_data

    Arguments:
        text (_Text): instance of class _Text

    Attributes:
        marked_up_n_grams (dict of dict): Dictionary with key of an n-gram and value of an n_gram_dict which
        represents frequency and range of an n-gram
        vocabulary_by_level_dict (dict): amount of _tokens by CEFR level

    Methods:
        word_freq_range
        n_gram_freq_range
        n_gram_proportion
        n_gram_accuracy
        academic_formulas_freq
        academic_vocabulary_content
        vocabulary_by_level
        get_full_data

    """

    def __init__(self, text=None, token_list=None, pos_tag=None):
        """
        The input can be both Text's instance or a list of _tokens. The last one is used for replacement options,
        and it takes as input list with _tokens and their part of speech tag (because they are synonyms they have
        similar pos tags)

        :param _Text text: If given than it is for text's lexical sophistication measurements
        :param list token_list: If given than it is for TokenReplacementOptions
        :param str pos_tag: Part of speech tag for tokens from a given token_list
        """
        if text is not None:
            assert isinstance(text, _Text)
            self._content = text.content
            self._tokens = text.get_tokens(
                include_punct=True, include_content_words=True, include_functional_words=True
            )
            self._content_tokens = text.get_tokens(
                include_punct=False, include_content_words=True, include_functional_words=False
            )
            self._functional_tokens = text.get_tokens(
                include_punct=False, include_content_words=False, include_functional_words=True
            )
            self._bigrams = text.get_bigrams()
            self._trigrams = text.get_trigrams()
        # For replacement options
        elif token_list is not None:
            assert isinstance(token_list, list)
            doc = Doc(nlp.vocab, words=token_list, pos=[pos_tag for i in range(len(token_list))], lemmas=token_list)
            banned_pos = ['PROPN', 'SYM', 'PART', 'CCONJ', 'ADP', 'X']
            self._tokens = [token for token in doc if (not token.is_punct)]
            self._content_tokens = [token for token in doc if (not token.is_stop)
                                    and (token.pos_ not in banned_pos) and (not token.is_punct)]
            self._functional_tokens = [token for token in doc if (token.is_stop or token.pos_ in banned_pos)
                                       and (not token.is_punct)]

        # Dictionary consisting of token and token_dict.
        # Token dict is a dictionary with frequency, range, academic and level keys and unique id
        self._marked_up_tokens = dict()
        for token in self._tokens:
            if token in self._functional_tokens:
                self._marked_up_tokens[token] = {
                    'punct': False, 'functional_word': True, 'id': self._tokens.index(token)
                }
            elif token in self._content_tokens:
                self._marked_up_tokens[token] = {
                    'punct': False, 'functional_word': False, 'freq': 0, 'range': 0,
                    'academic': bool(), 'level': None, 'id': self._tokens.index(token)
                }
            else:
                self._marked_up_tokens[token] = {
                    'punct': True, 'functional_word': False, 'id': self._tokens.index(token)
                }

        # Dictionary consisting of n-gram and n-gram_dict.
        # N-gram dict is a dictionary with frequency and range.
        # Dictionary includes academic formulas marked with frequency and occurrences in a text as well
        self.marked_up_n_grams = dict()

        # Lists consisting of bigrams and trigrams which are unrecognized in corpus. Used for n-gram proportion
        self._unrecognized_bigrams = list()
        self._unrecognized_trigrams = list()

        # Dictionary with keys as CEFR levels (A1, A2, B1, etc.) and appropriate amount of _tokens in a text
        self.vocabulary_by_level_dict = dict()

    def word_freq_range(self, for_replacement_options=False):
        """
        Add frequency and range of a token to self._marked_up_tokens.
        Calculates average word frequency & range for all words as well as for content and functional words separately.

        Returns: dict containing average word frequency & range for all words as well as for content and
        functional words separately.

        :param bool for_replacement_options: Only marks up tokens with CEFR level without return
        """
        # Load and read tagged corpus, then tokenize (and lemmatize) it and create a frequency and range dictionary
        brown_freq = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/tagged_brown', verbose=False))
        )
        brown_range = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/tagged_brown', verbose=False)), calc='range'
        )

        # Create lists of frequencies and ranges for all words (AW), content words (CW) and functional words (FW)
        # to calculate average frequency and range
        all_words_frequencies, all_words_ranges = list(), list()
        content_words_frequencies, content_words_ranges = list(), list()
        functional_words_frequencies, functional_words_ranges = list(), list()
        for token in self._tokens:
            try:
                word = f'{token.lemma_}_{token.pos_}'.lower()
                word_freq = brown_freq[word]
                word_range = brown_range[word]
                self._marked_up_tokens[token]['freq'] = word_freq
                self._marked_up_tokens[token]['range'] = word_range


                all_words_frequencies.append(word_freq)
                all_words_ranges.append(word_range)
                if self._marked_up_tokens[token]['functional_word']:
                    functional_words_frequencies.append(word_freq)
                    functional_words_ranges.append(word_range)
                else:
                    content_words_frequencies.append(word_freq)
                    content_words_ranges.append(word_range)
            except KeyError:
                # Ignores KeyError that occurs due to the absence of a token in the corpus
                continue

        if not for_replacement_options:
            measurements_dict = {
                'All words frequency': np.mean(all_words_frequencies),
                'All words range': np.mean(all_words_ranges),
                'Content words frequency': np.mean(content_words_frequencies),
                'Content words range': np.mean(content_words_ranges),
                'Functional words frequency': np.mean(functional_words_frequencies),
                'Functional words range': np.mean(functional_words_ranges)
            }

            return measurements_dict

    def n_gram_freq_range(self):
        """
        Create lists with bi-, trigrams and its frequency & range.
        Calculates average bi-, trigrams frequency & range.

        Returns: dict containing average frequency and range.

        """
        # Create n-grams frequency and range dictionary
        brown_bigram_freq = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=2)
        )
        brown_bigram_range = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=2),
            calc='range'
        )
        brown_trigram_freq = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=3)
        )
        brown_trigram_range = ct.frequency(ct.tokenize(ct.ldcorpus(
            'corpora/brown', verbose=False), lemma=False, ngram=3),
            calc='range'
        )

        # Create lists of frequencies and ranges for calculating average values further
        bigram_frequencies, bigram_ranges = list(), list()
        trigram_frequencies, trigram_ranges = list(), list()
        for bigram in self._bigrams:
            try:
                bigram_freq = brown_bigram_freq[bigram]
                bigram_range = brown_bigram_range[bigram]

                bigram_frequencies.append(bigram_freq)
                bigram_ranges.append(bigram_range)

                # Add bigram marked with frequency and range to a dict
                self.marked_up_n_grams[bigram] = {'freq': bigram_freq, 'range': bigram_range, 'academic': False,
                                                  'len': 2}
            except KeyError:
                # Unrecognized bigrams for bigram proportion
                self._unrecognized_bigrams.append(bigram)
                continue
        for trigram in self._trigrams:
            try:
                trigram_freq = brown_trigram_freq[trigram]
                trigram_range = brown_trigram_range[trigram]

                trigram_frequencies.append(trigram_freq)
                trigram_ranges.append(trigram_range)

                # Add trigram marked with frequency and range to a dict
                self.marked_up_n_grams[trigram] = {'freq': trigram_freq, 'range': trigram_range, 'academic': False,
                                                   'len': 3}
            except KeyError:
                # Unrecognized trigrams for trigram proportion
                self._unrecognized_trigrams.append(trigram)
                continue

        measurements_dict = {
            'Bigram frequency': np.mean(bigram_frequencies),
            'Bigram range': np.mean(bigram_ranges),
            'Trigram frequency': np.mean(trigram_frequencies),
            'Trigram range': np.mean(trigram_ranges)
        }

        return measurements_dict

    def n_gram_proportion(self):
        """
        Proportion is unrecognized (not found in corpus) n-grams to all n-grams ratio.

        Returns: dict with bigram and trigram proportion.

        """
        # Check if n-grams are not marked up
        if len(self.marked_up_n_grams) == 0:
            self.n_gram_freq_range()

        measurements_dict = {
            'Bigram proportion': len(self._unrecognized_bigrams) / len(self._bigrams),
            'Trigram proportion': len(self._unrecognized_trigrams) / len(self._trigrams)
        }

        return measurements_dict

    def n_gram_accuracy(self):
        """
        Accuracy is a correlation between bigrams/trigrams normalized frequency in an essay and
        bigrams/trigrams normalized frequency in a corpus. It should be noted that we consider only bigrams and trigrams
        which are represented both in an essay and a corpus.

        By bigram/trigram normalized frequency in an essay we mean  number of occurrences divided by
        number of all bigrams/trigrams respectfully. The same for bigram/trigram normalized frequency in corpus.

        Therefore, by calculating bigram and trigram accuracy we want to know how similar n-gram using in an essay
        comparing to using in corpus.

        Returns: dict with bigram and trigram accuracy.

        """
        # Check if n-grams are not marked up
        if len(self.marked_up_n_grams) == 0:
            self.n_gram_freq_range()

        # Create lists of bigrams and trigrams from an essay which are represented in corpus
        trigrams = list([trigram for trigram in self._trigrams if trigram in self.marked_up_n_grams.keys()])
        bigrams = list([bigram for bigram in self._bigrams if bigram in self.marked_up_n_grams.keys()])

        # Create lists of bigrams and trigrams from a corpus
        brown_bigrams = list(ct.tokenize(ct.ldcorpus('corpora/brown', verbose=False), lemma=False, ngram=2))
        brown_trigrams = list(ct.tokenize(ct.ldcorpus('corpora/brown', verbose=False), lemma=False, ngram=3))

        # Create lists of bigrams and trigrams normalized frequency from an essay and from a corpus
        bigrams_normalized_frequency_essay = list()
        bigrams_normalized_frequency_corpus = list()
        trigrams_normalized_frequency_essay = list()
        trigrams_normalized_frequency_corpus = list()
        for n_gram, n_gram_dict in self.marked_up_n_grams.items():
            if n_gram_dict['len'] == 2:
                normalized_frequency = bigrams.count(n_gram) / len(bigrams)
                bigrams_normalized_frequency_essay.append(normalized_frequency)

                normalized_frequency = n_gram_dict['freq'] / len(brown_bigrams)
                bigrams_normalized_frequency_corpus.append(normalized_frequency)
            else:
                normalized_frequency = trigrams.count(n_gram) / len(trigrams)
                trigrams_normalized_frequency_essay.append(normalized_frequency)

                normalized_frequency = n_gram_dict['freq'] / len(brown_trigrams)
                trigrams_normalized_frequency_corpus.append(normalized_frequency)

        # Calculate correlation (accuracy)
        bigram_accuracy, bigram_p = stats.pearsonr(
            bigrams_normalized_frequency_corpus, bigrams_normalized_frequency_essay
        )
        trigram_accuracy, trigram_p = stats.pearsonr(
            trigrams_normalized_frequency_corpus, trigrams_normalized_frequency_essay
        )

        if np.isnan(bigram_accuracy):
            bigram_accuracy = 0
        if np.isnan(trigram_accuracy):
            trigram_accuracy = 0

        measurements_dict = {
            'Bigram accuracy': bigram_accuracy,
            'Trigram accuracy': trigram_accuracy
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
            if formula[0] in self._content:
                frequencies += int(formula[1]) * self._content.count(formula[0])
                occurrences += self._content.count(formula[0])

                self.marked_up_n_grams[formula[0]] = {
                    'freq': int(formula[1]), 'range': None, 'occur': self._content.count(formula[0]), 'academic': True,
                    'len': len(formula[0].split(' '))
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
        between the number of academic words and the number of _tokens (not counting stopwords).

        Returns: dictionary with amount of academic words and their percentage of the text.

        """
        academic_word_list = list()
        with open('corpora/AWL.txt', 'r') as file:
            for row in file:
                academic_word_list.append(row.rstrip('\n'))

        academic_words = list()
        for token in self._content_tokens:
            if token.text in academic_word_list:
                self._marked_up_tokens[token]['academic'] = True
                academic_words.append(token)

        statistics_dict = {
            'Amount of academic words': len(academic_words),
            'Percentage of academic words': len(academic_words) / len(self._content_tokens)
        }

        return statistics_dict

    def vocabulary_by_level(self, for_replacement_options=False):
        """
        Marks up tokens with CEFR level (A1, A2, B1, etc.) and calculates amount of tokens by each CEFR level.

        Returns a dictionary.

        :param bool for_replacement_options: Only marks up tokens with CEFR level without return
        """
        levels = {
            2: 'A1', 3: 'A2', 4: 'B1', 5: 'B2', 6: 'C1'
        }
        efllex_corpus = pd.read_csv(
            'corpora/EFLLex_NLP4J',
            delimiter='\\t',
            engine='python'
        )
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

        for token in self._content_tokens:
            if type(tags_dict[token.pos_]) is str:
                # Some words may not be in corpus, therefore we will use try/except
                try:
                    level = tokens_with_CEFR_level_corpus[(token.lemma_, tags_dict[token.pos_])]
                    tokens_by_CEFR_level[level].append(token)
                    self._marked_up_tokens[token]['level'] = level
                except KeyError:
                    self._marked_up_tokens[token]['level'] = 'C2'
                    tokens_by_CEFR_level['C2'].append(token)
            # If there are several tags for one spaCy tag we try all of them
            elif type(tags_dict[token.pos_]) is list:
                for tag_option in tags_dict[token.pos_]:
                    try:
                        level = tokens_with_CEFR_level_corpus[(token.lemma_, tag_option)]
                        tokens_by_CEFR_level[level].append(token)
                        self._marked_up_tokens[token]['level'] = level
                    except KeyError:
                        pass
                if self._marked_up_tokens[token]['level'] is None:
                    self._marked_up_tokens[token]['level'] = 'C2'
                    tokens_by_CEFR_level['C2'].append(token)
            elif tags_dict[token.pos_] is None:
                continue
            else:
                print(f'Token {token.text} has unknown part of speech tag that is {token.pos_}')

        self.vocabulary_by_level_dict = {
            'A1 words': len(tokens_by_CEFR_level['A1']), 'A2 words': len(tokens_by_CEFR_level['A2']),
            'B1 words': len(tokens_by_CEFR_level['B1']), 'B2 words': len(tokens_by_CEFR_level['B2']),
            'C1 words': len(tokens_by_CEFR_level['C1']), 'C2 words': len(tokens_by_CEFR_level['C2']),
        }

        level_weight = {
            'A1': 1, 'A2': 1, 'B1': 2, 'B2': 4, 'C1': 8, 'C2': 16
        }

        vocabulary_metric = list()
        for level in tokens_by_CEFR_level.keys():
            for i in range(self.vocabulary_by_level_dict[f'{level} words']):
                vocabulary_metric.append(i * level_weight[level])

        if not for_replacement_options:
            vocabulary_metric_dict = {
                'Vocabulary': np.mean(vocabulary_metric)
            }

            return vocabulary_metric_dict

    def get_full_data(self):
        """
        Combines data from all methods into one dictionary.

        Returns: dictionary with data from all methods
        """
        word_freq_range_data = self.word_freq_range()
        n_gram_freq_range_data = self.n_gram_freq_range()
        n_gram_proportion = self.n_gram_proportion()
        n_gram_accuracy = self.n_gram_accuracy()
        academic_vocabulary_data = self.academic_vocabulary_content()
        academic_formulas_data = self.academic_formulas_freq()
        vocabulary_by_level_data = self.vocabulary_by_level()

        full_data = (word_freq_range_data | n_gram_freq_range_data | n_gram_proportion | n_gram_accuracy \
                     | academic_vocabulary_data | academic_formulas_data | vocabulary_by_level_data)

        return full_data

    def get_marked_up_tokens(self, include_functional_words=True):
        """
        Dictionary with key of a token and value of a token_dict which represents characteristics of a token
        (stopword, frequency, range, academic, level CEFR)

        :param bool include_functional_words: Whether include stopwords or not (for TokenReplacementOptions)
        """
        if include_functional_words:
            return self._marked_up_tokens
        else:
            return {key: value for key, value in self._marked_up_tokens.items()
                    if (value['functional_word'] is False) and (value['punct'] is False)}


class _LexicalDiversityMeasurements:
    """
    Class measures lexical diversity using TTR, MTLD and MTLD MA Wrap (MTLD-W) indices.
    Returns a dictionary with measurements.

    Arguments:
        text (_Text): instance of class _Text

    Methods:
        indices_data

    """

    def __init__(self, text):
        assert isinstance(text, _Text)
        self._lemmatized_text = text.get_lemmas(include_functional_words=False)

    def indices_data(self):
        data_dict = {
            'TTR': ld.ttr(self._lemmatized_text),
            'Root TTR': ld.root_ttr(self._lemmatized_text),
            'Log TTR': ld.log_ttr(self._lemmatized_text),
            'Maas TTR': ld.maas_ttr(self._lemmatized_text),
            'D': ld.hdd(self._lemmatized_text),
            'MTLD': ld.mtld(self._lemmatized_text),
            'MTLD MA Wrap': ld.mtld_ma_wrap(self._lemmatized_text)
        }
        return data_dict


class TokensAreNotRecognized(Exception):
    pass


class TextAnalysis:
    """
    Analysis of a given essay by lexical diversity and lexical sophistication.
    Provides tables with trigrams with the biggest frequency/range, lexical diversity indices data, academic formulas
    and a pie chart with the CEFR levels (A1, A2, B1, etc.) and appropriate number of words from a given essay.

    Arguments:
        essay (str): Essay presented as string

    Methods:
        get_data_for_web

    """

    def __init__(self, essay):
        assert isinstance(essay, str)
        self._text = _Text(essay)
        self._lex_div = _LexicalDiversityMeasurements(self._text)
        self._lex_sop = _LexicalSophisticationMeasurements(self._text)

        self._lexical_sophistication_measurements = self._lex_sop.get_full_data()
        self._lexical_diversity_measurements = self._lex_div.indices_data()
        self._marked_up_tokens = self._lex_sop.get_marked_up_tokens()
        self._marked_up_n_grams = self._lex_sop.marked_up_n_grams
        self._vocabulary_by_level_dict = self._lex_sop.vocabulary_by_level_dict

    def get_vocabulary_chart(self):
        """
        Creates a pie chart with the CEFR levels (A1, A2, B1, etc.) and appropriate number of words from a given essay.

        The result is vocabulary_chart.png file
        """
        fig, ax = plt.subplots(facecolor=(0.1, 0.2, 0.5, 0))

        # Amount of vocabulary for each level
        amount_of_vocabulary_by_level = list(self._vocabulary_by_level_dict.values())
        # Levels (labels)
        levels_of_vocabulary = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        colors = ['#FFE89C', '#FFCF32', '#EA8198',
                  '#EC2C55', '#AB60C6', '#7E00AC']

        ax.pie(amount_of_vocabulary_by_level,
               labels=levels_of_vocabulary,
               autopct='%1.1f%%',
               colors=colors
               )
        plt.savefig('temporary_files/vocabulary_chart.png')

    def get_trigrams_dataframe(self):
        """
        Creates a pandas DataFrame with trigrams with the biggest frequency or range from a given essay.

        Returns a pandas DataFrame
        """
        sorted_trigrams = sorted(self._marked_up_n_grams.items(), key=lambda x: x[1]['freq'], reverse=True)
        # Dictionary that will be converted into CSV table
        trigrams_dict = {
            'Trigram': list(), 'Frequency': list(), 'Range': list()
        }
        for trigram_tuple in sorted_trigrams:
            if trigram_tuple[1]['len'] == 3:
                trigrams_dict['Trigram'].append(trigram_tuple[0].replace('__', ' '))
                trigrams_dict['Frequency'].append(trigram_tuple[1]['freq'])
                trigrams_dict['Range'].append(trigram_tuple[1]['range'])
        trigrams = pd.DataFrame(trigrams_dict)
        return trigrams

    def get_academic_formulas_dataframe(self):
        """
        Creates a pandas DataFrame with academic formulas and their frequency from a given text.

        Returns a pandas DataFrame
        """
        # Dictionary that will be converted into CSV table
        academic_formulas_dict = {
            'Academic formula': list(), 'Frequency': list()
        }
        for n_gram, n_gram_dict in self._marked_up_n_grams.items():
            if n_gram_dict['academic'] is True:
                academic_formulas_dict['Academic formula'].append(n_gram)
                academic_formulas_dict['Frequency'].append(n_gram_dict['freq'])
        academic_formulas = pd.DataFrame(academic_formulas_dict)

        if academic_formulas.empty:
            academic_formulas = pd.DataFrame({'Academic formula': ['Not found'], 'Frequency': ['-']})

        return academic_formulas

    def get_stats_dataframe(self):
        """
        Creates a csv table with data from indices and percentage of academic words.

        Returns a csv table
        """
        stats_dict = {'metric': list(), 'value': list()}
        stats_dict['metric'].append('TTR')
        stats_dict['value'].append(round(self._lexical_diversity_measurements['TTR'], 2))
        stats_dict['metric'].append('Academic words')
        stats_dict['value'].append(str(self._lexical_sophistication_measurements['Amount of academic words']))
        stats_dict['metric'].append('Average trigram frequency')
        stats_dict['value'].append(round(self._lexical_sophistication_measurements['Trigram frequency'], 2))
        stats = pd.DataFrame(stats_dict)
        return stats

    def get_recurring_lemmas_dataframe(self, include_functional_words=False):
        """
        Returns a pandas DataFrame of lemmas which occur in a text 2 or more times

        :param bool include_functional_words: Include stopwords, proper nouns, symbols, particles, adposition,
                                           coordinating conjunction and unknown parts of speech
        """
        recurring_lemmas_dict = {
            'Lemma': list(), 'Occurrences': list()
        }
        if include_functional_words:
            lemmas_occurrences = self._text.get_lemmas_occurrences(include_functional_words=True)
        else:
            lemmas_occurrences = self._text.get_lemmas_occurrences(include_functional_words=False)

        for lemma, occurrences in lemmas_occurrences.items():
            if occurrences >= 2:
                recurring_lemmas_dict['Lemma'].append(lemma)
                recurring_lemmas_dict['Occurrences'].append(occurrences)
        recurring_lemmas = pd.DataFrame(recurring_lemmas_dict).sort_values(by='Occurrences', ascending=False)

        return recurring_lemmas

    def get_level(self):
        """Returns a CEFR level (A1, A2, B1, etc.) based on TTR value as the metric with the biggest correlation"""
        TTR = self._lexical_diversity_measurements['TTR']
        coefficients = [-59.4178015, 69.99997391, -13.2177359]

        degree = len(coefficients) - 1
        score = sum([coefficients[i] * (TTR ** (degree - i)) for i in range(len(coefficients))])
        if score <= 4:
            return f'A2 ({round(score)}/9)'
        elif score <= 5:
            return f'B1 ({round(score)}/9)'
        elif score <= 6.5:
            return f'B2 ({round(score)}/9)'
        elif score <= 8:
            return f'C1 ({round(score)}/9)'
        else:
            return f'C2 ({round(score)}/9)'

    def _get_data_for_web(self):
        """
        Returns table with the most frequent trigrams, stats (indices and other information), academic formulas
        and saves vocabulary_chart.png
        """
        trigrams = self.get_trigrams_dataframe()
        # Leave only first 10 entities or fewer
        if trigrams.shape[0] >= 10:
            trigrams = trigrams.head(10)
        else:
            trigrams = trigrams.head(trigrams.shape[0])

        academic_formulas = self.get_academic_formulas_dataframe()
        # Leave only first 10 entities or fewer
        if academic_formulas.shape[0] >= 10:
            academic_formulas = academic_formulas.head(10)
        else:
            academic_formulas = academic_formulas.head(academic_formulas.shape[0])

        recurring_lemmas = self.get_recurring_lemmas_dataframe()
        # Leave only first 10 entities or fewer
        if recurring_lemmas.shape[0] >= 10:
            recurring_lemmas = recurring_lemmas.head(10)
        else:
            recurring_lemmas = recurring_lemmas.head(recurring_lemmas.shape[0])

        level = self.get_level()
        stats = self.get_stats_dataframe()
        self.get_vocabulary_chart()
        return trigrams, stats, academic_formulas, recurring_lemmas, level

    @property
    def lexical_sophistication_measurements(self):
        return self._lexical_sophistication_measurements

    @property
    def lexical_diversity_measurements(self):
        return self._lexical_diversity_measurements

    @property
    def marked_up_tokens(self):
        return self._marked_up_tokens

    @property
    def marked_up_n_grams(self):
        return self._marked_up_n_grams


class TokenReplacementOptions:
    """
    Selects synonyms with lower frequency or lower range or higher CEFR level to a given token.
    It should be mentioned that suggested synonyms might be inappropriate in a text because of different semantics
    that does not count.

    Arguments:
        marked_up_tokens (dict): Dictionary of _tokens from a text marked up with level, frequency and range

    Methods:
        get_replacement_options: Returns replacement options based on vocabulary level, frequency and range

    """

    def __init__(self, marked_up_tokens):
        # Assigns dictionary of _tokens from a text marked up with level, frequency and range
        assert isinstance(marked_up_tokens, dict)
        self._marked_up_tokens = marked_up_tokens

    @staticmethod
    def __spacy_pos_to_wordnet_pos(spacy_pos):
        """
        Transforms spaCy part of speech tag to wordnet pos tag

        :param str spacy_pos: Part of speech in spaCy format
        """
        try:
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
        # For errors on WordNet side
        except AttributeError:
            return None

    def __get_synonyms(self, token):
        """
        Returns synonyms of a token based on part of speech

        :param spacy.tokens.token.Token token: SpaCy token from a text
        """
        assert isinstance(token, spacy.tokens.token.Token)
        pos_tag = self.__spacy_pos_to_wordnet_pos(token.pos_)
        synonyms = list()
        for synset in wn.synsets(token.text, pos=pos_tag):
            # synonym = synset.lemmas()[0].name()
            # if synonym not in synonyms:
            #     synonyms.append(synonym)
            for synonym in synset.lemmas():
                if synonym.name() not in synonyms:
                    synonyms.append(synonym.name())
        return synonyms

    def get_replacement_options(self, token, return_token_text=False):
        """
        Returns replacement options based on synonyms of a token excluding synonyms with lower CEFR level,
        higher frequency & range and ones which are not in EFLLex corpus.

        :param spacy.tokens.token.Token token: SpaCy token from a text
        :param bool return_token_text: Whether return text of tokens (str format) or tokens (spaCy token format)
        """
        assert isinstance(token, spacy.tokens.token.Token)
        # Assigns instance of LexicalSophisticationMeasurements for synonyms
        lexical_sophistication = _LexicalSophisticationMeasurements(token_list=self.__get_synonyms(token),
                                                                    pos_tag=token.pos_)
        # Marks up synonymic _tokens with frequency and range
        lexical_sophistication.word_freq_range(for_replacement_options=True)
        # Marks up synonymic _tokens with level
        lexical_sophistication.vocabulary_by_level(for_replacement_options=True)
        marked_up_synonyms = lexical_sophistication.get_marked_up_tokens(include_functional_words=False)

        # For now, we only imagine that we have token level, frequency and range
        token_level = self._marked_up_tokens[token]['level']
        token_freq = self._marked_up_tokens[token]['freq']
        token_range = self._marked_up_tokens[token]['range']
        # Excludes synonyms which are not in EFLLex corpus
        synonyms = {key: value for key, value in marked_up_synonyms.items() if value['level'] != 'C2'}
        # Leaves only synonyms with higher CEFR level or lower frequency or lower range and excludes freq/range equal 0
        levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        synonyms = {key: value for key, value in synonyms.items()
                    if (value['freq'] < token_freq) or (value['range'] < token_range)
                    or (levels.index(value['level']) > levels.index(token_level))
                    and ((value['freq'] != 0) and (value['range'] != 0))}
        # Sorts synonyms by level
        synonyms_sorted_by_level = sorted(synonyms.items(), key=lambda x: x[1]['level'], reverse=True)
        if return_token_text:
            replacements = [synonym_set[0].text for synonym_set in synonyms_sorted_by_level]
        else:
            replacements = [synonym_set[0] for synonym_set in synonyms_sorted_by_level]
        return replacements[0:2]


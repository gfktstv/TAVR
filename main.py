import spacy
from spacy_ngram import NgramComponent

from nltk.corpus import wordnet as wn

from lexical_diversity import lex_div as ld
from corpus_toolkit import corpus_tools as ct

from collections import Counter

from math import log
from statistics import mean

import pandas as pd

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
        self.source = source
        with open(f'{self.source}', 'r') as file:
            self.content = file.read().replace('\n', ' ')

        self.nlp_doc = nlp(self.content)

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

    def __str__(self):
        return text.content


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
        marked_up_n_grams (dict of dict): Dictionary with key of a n-gram and value of a n_gram_dict which
        represents frequency and range of an n-gram

        brown_freq (dict): Brown single corpus of tokens marked with frequency
        brown_range (dict): brown single corpus of tokens marked with range
        bigram_freq (dict): brown single corpus of bigrams marked with frequency
        bigram_range (dict): brown single corpus of bigrams marked with range
        trigram_freq(dict): brown single corpus of trigrams marked with frequency
        trigram_range (dict): brown single corpus of trigrams marked with range
        bigrams_freq_range (list): list of bigrams from corpus marked with frequency and range
        trigrams_freq_range (list): list of trigrams from corpus marked with frequency and range


    Methods:
        word_freq_range
        n_gram_freq_range
        academic_formulas_freq
        academic_vocabulary_content
        vocabulary_by_level
        word_specificity

    """

    def __init__(self, text_class_variable):
        # Attributes of variable of class Text
        self.content = text_class_variable.content
        self.tokens = text_class_variable.get_tokens()
        self.significant_tokens = text_class_variable.get_tokens(False, False)
        self.bigrams = text_class_variable.get_bigrams()
        self.trigrams = text_class_variable.get_trigrams()

        # Dictionary consisting of token and token_dict.
        # Token dict is a dictionary with frequency, range, academic and level keys,
        # which reflect frequency and range of the token, being in AWL and its level
        self.marked_up_tokens = dict()
        for token in self.tokens:
            if not token.is_stop:
                self.marked_up_tokens[token] = {
                    'stopword': False, 'freq': None, 'range': None,
                    'academic': bool(), 'level': None
                }
            if token.is_stop:
                self.marked_up_tokens[token] = {'stopword': True}

        # Dicts for brown single corpora marked with frequency and range
        self.brown_freq = dict()
        self.brown_range = dict()

        # Dicts for brown single corpora containing bigrams and trigrams marked with frequency and range
        self.bigram_freq = dict()
        self.bigram_range = dict()
        self.trigram_freq = dict()
        self.trigram_range = dict()

        # List of bigrams and trigrams from a content marked with frequency and range (including logarithmic scores)
        self.bigrams_freq_range = list()
        self.trigrams_freq_range = list()

        # Academic Word List
        self.academic_word_list = list()
        with open('academic word list/AWL.txt', 'r') as file:
            for row in file:
                self.academic_word_list.append(row.rstrip('\n'))

        # Academic Formulas list
        AFL_csv = pd.read_html('https://www.eapfoundation.com/vocab/academic/afl/')[5]
        AFL_csv = AFL_csv.drop([0])[[1, 3]]
        AFL_csv.columns = ['Formula', 'Frequency per million']
        self.academic_formulas_list = [
            [record['Formula'], record['Frequency per million']] for record in AFL_csv.to_dict('records')
        ]

        # Dict of academic formulas from a content with frequency and amount of occurrences
        self.content_academic_formulas = dict()

        # Dict of tokens marked with CEFR level from corpus
        self.tokens_with_CEFR_level_corpus = {}
        levels = {
            2: 'A1', 3: 'A2', 4: 'B1', 5: 'B2', 6: 'C1'
        }
        efllex_corpus = pd.read_csv('EFLLex_NLP4J', delimiter='\\t', engine='python')
        for i in range(len(efllex_corpus.iloc[:, 0])):
            for level in [2, 3, 4, 5, 6]:
                level_frequency = efllex_corpus.iloc[i, level]
                if level_frequency > 0:  # Selects first level where the token appears
                    # The key in dictionary is tuple with token lemma, and it's part of speech tag,
                    # the value is it's level
                    lemma_and_pos = (efllex_corpus.iloc[i, 0], efllex_corpus.iloc[i, 1])
                    self.tokens_with_CEFR_level_corpus[lemma_and_pos] = levels[level]
                    break

    def word_freq_range(self):
        """
        Add frequency and range of a token to self.marked_up_tokens.
        Calculates average word frequency & range (including logarithmic scores).

        Returns: dict containing average frequency and range.

        """
        # Load and read tagged corpus, then tokenize (and lemmatize) it and create a frequency and range dictionary
        self.brown_freq = ct.frequency(ct.tokenize(ct.ldcorpus('tagged_brown', verbose=False)))
        self.brown_range = ct.frequency(ct.tokenize(ct.ldcorpus('tagged_brown', verbose=False)), calc='range')

        # Create lists of frequencies and ranges for calculating average values further
        word_frequencies, word_ranges = list(), list()
        for token in self.significant_tokens:
            try:
                word = f'{token.lemma_}_{token.pos_}'.lower()
                word_freq = self.brown_freq[word]
                word_range = self.brown_range[word]
                self.marked_up_tokens[token]['freq'] = word_freq
                self.marked_up_tokens[token]['range'] = word_range

                word_frequencies.append(word_freq)
                word_ranges.append(word_range)
            except KeyError:
                # Ignores KeyError that occurs due to the absence of a token in the corpus
                continue

        measurements_dict = {
            'Word frequency average: ': mean(word_frequencies),
            'Word range average: ': mean(word_ranges),
            'Word logarithmic frequency average: ': mean(log(x) for x in word_frequencies),
            'Word logarithmic range average: ': mean(log(x) for x in word_frequencies)
        }

        return measurements_dict

    def n_gram_freq_range(self):
        """
        Create lists with bi-, trigrams and its frequency & range.
        Calculates average bi-, trigrams frequency & range (including logarithmic scores).

        Returns:
            1) dict containing average frequency and range (including logarithmic scores),
            2) list of bigrams marked with frequency and range,
            3) list of trigrams marked with frequency and range.


        """
        # Create n-grams frequency and range dictionary
        self.bigram_freq = ct.frequency(ct.tokenize(ct.ldcorpus('brown', verbose=False), lemma=False, ngram=2))
        self.bigram_range = ct.frequency(
            ct.tokenize(ct.ldcorpus("brown", verbose=False), lemma=False, ngram=2), calc='range'
        )
        self.trigram_freq = ct.frequency(ct.tokenize(ct.ldcorpus('brown', verbose=False), lemma=False, ngram=3))
        self.trigram_range = ct.frequency(ct.tokenize(ct.ldcorpus(
            "brown", verbose=False), lemma=False, ngram=3), calc='range'
        )

        # Create lists of frequencies and ranges for calculating average values further
        bigram_frequencies, bigram_ranges = list(), list()
        trigram_frequencies, trigram_ranges = list(), list()
        for bigram in self.bigrams:
            try:
                bigram_freq = self.bigram_freq[bigram]
                bigram_range = self.bigram_range[bigram]

                bigram_frequencies.append(bigram_freq)
                bigram_ranges.append(bigram_range)

                # Append list of bigram and its frequency & range for marking content with high frequency bigrams
                self.bigrams_freq_range.append([bigram, bigram_freq, bigram_range])
            except KeyError:
                self.bigrams_freq_range.append([bigram, 'not_found'])
        for trigram in self.trigrams:
            try:
                trigram_freq = self.trigram_freq[trigram]
                trigram_range = self.trigram_range[trigram]

                trigram_frequencies.append(trigram_freq)
                trigram_ranges.append(trigram_range)

                # Append list of trigram and its frequency & range for marking content with high frequency trigrams
                self.trigrams_freq_range.append([trigram, trigram_freq, trigram_freq])
            except KeyError:
                self.trigrams_freq_range.append([trigram, 'not_found'])

        measurements_dict = {
            'Bigram frequency average': mean(bigram_frequencies),
            'Bigram range average': mean(bigram_ranges),
            'Bigram logarithmic frequency average': mean(log(x) for x in bigram_frequencies),
            'Bigram logarithmic range average': mean(log(x) for x in bigram_ranges),

            'Trigram frequency average': mean(trigram_frequencies),
            'Trigram range average': mean(trigram_ranges),
            'Trigram logarithmic frequency average': mean(log(x) for x in trigram_frequencies),
            'Trigram logarithmic range average': mean(log(x) for x in trigram_ranges)
        }

        return measurements_dict, self.bigrams_freq_range, self.trigrams_freq_range

    def academic_formulas_freq(self):
        """
        Create list with academic formulas and their frequency (including logarithmic) and amount of occurrences.
        Calculates average frequency (including logarithmic).

        Returns:
            1) list of academic frequencies marked with frequency (including logarithmic),
            2) dict containing average frequency (including logarithmic).

        """
        frequencies = 0
        occurrences = 0
        for formula in self.academic_formulas_list:
            if formula[0] in self.content:
                frequencies += int(formula[1]) * self.content.count(formula[0])
                occurrences += self.content.count(formula[0])

                self.content_academic_formulas[formula[0]] = {
                    'Frequency': int(formula[1]), 'Occurrences': self.content.count(formula[0])
                }

        if occurrences < 3:
            statistics_dict = {
                'Academic formulas frequency': None,
                'Academic formulas logarithmic frequency': None
            }
        else:
            statistics_dict = {
                'Academic formulas frequency': frequencies / occurrences,
                'Academic formulas logarithmic frequency': log(frequencies / occurrences)
            }

        return statistics_dict, self.content_academic_formulas

    def academic_vocabulary_content(self):
        """
        Calculates amount of academic words (words from Academic Word List) and the ratio
        between the number of academic words and the number of tokens (not counting stopwords).

        Returns: dictionary with amount of academic words and their percentage of the text.

        """
        academic_words = list()
        for token in self.significant_tokens:
            if token.text in self.academic_word_list:
                self.marked_up_tokens[token]['academic'] = True
                academic_words.append(token)

        statistics_dict = {
            'Amount of academic words': len(academic_words),
            'Percentage of academic words': len(academic_words) / len(self.significant_tokens)
        }

        return statistics_dict

    def vocabulary_by_level(self):
        """
        Calculates amount of tokens for each CEFR level using 'words by level' dir and
        percentage of words for each category (further visualized by a pie plot).

        Returns: dictionary with amount of words for each CEFR level, their percentage of the text.

        """
        tokens_by_CEFR_level = {
            'A1': list(), 'A2': list(),
            'B1': list(), 'B2': list(),
            'C1': list(), 'C2': list()
        }
        # Dict to convert spaCy's tags to the corpus' tags. Some of tags are empty (None) because they are
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
            if type(tags_dict[token.pos_]) == str:
                # Some words may not be in corpus, therefore we will use try/except
                try:
                    level = self.tokens_with_CEFR_level_corpus[(token.lemma_, tags_dict[token.pos_])]
                    tokens_by_CEFR_level[level].append(token)
                    self.marked_up_tokens[token]['level'] = level
                except KeyError:
                    self.marked_up_tokens[token]['level'] = 'C2'
                    tokens_by_CEFR_level['C2'].append(token)
            # If there are several tags for one spaCy tag we try all of them
            elif type(tags_dict[token.pos_]) == list:
                for tag_option in tags_dict[token.pos_]:
                    try:
                        level = self.tokens_with_CEFR_level_corpus[(token.lemma_, tag_option)]
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

        return vocabulary_by_level_dict

    def get_full_data(self):
        """
        Combines data from all methods except into one dictionary.

        Returns: dictionary with data from all methods
        """
        word_freq_range_data = self.word_freq_range()
        n_gram_freq_range_data, bigrams, trigrams = self.n_gram_freq_range()
        academic_vocabulary_data = self.academic_vocabulary_content()
        academic_formulas_data, formulas = self.academic_formulas_freq()
        vocabulary_by_level_data = self.vocabulary_by_level()

        full_data = word_freq_range_data | n_gram_freq_range_data | academic_vocabulary_data \
                    | academic_formulas_data | vocabulary_by_level_data

        return full_data


class LexicalDiversityMeasurement:
    """
    Class measures lexical diversity using TTR, MTLD and MTLD MA Wrap (MTLD-W) indexes.
    Returns a dictionary with measurements.

    Attributes:
        flemmatized_text (list): a content that is flemmatized by lexical diversity package

    Methods:
        indexes_data

    """

    def __init__(self, text_class_variable):
        self.flemmatized_text = ld.flemmatize(text_class_variable.content)

    def indexes_data(self):
        data_dict = {
            'TTR': ld.ttr(self.flemmatized_text),
            'MTLD': ld.mtld(self.flemmatized_text),
            'MTLD MA Wrap': ld.mtld_ma_wrap(self.flemmatized_text)
        }
        return data_dict


class SemanticRelationsMeasurement:
    """
    Class which measures vocabulary richness of the text by semantic relations.
    It uses an algorithm which create a paradigm of semantically related words
    from the text for each token and then extract indicators of those semantic lists.
    Returns a dictionary with statistics

    Attributes:
        tokens (set): Tokens without token.text duplicates
        sem_list (list): Paradigm of semantic related words to the one
        sem_lists (list): List of semantic lists

    Methods:
        statistics

    """

    def __init__(self, text_class_variable):
        self.content = text_class_variable.content
        self.tokens = text_class_variable.get_unique_tokens()
        self.sem_list = list()
        self.sem_lists = list()

    def statistics(self):
        """
        This algorithm extracts from text semantic lists - paradigm of words which semantically connected to the one.

        Description of the algorithm:
        1. For each token search semantically related words in the text (>0.65 similarity, but <0.8)
        2. Add each semantic list to a list of semantic lists
        3. Clear semantic lists:
        3.1. Delete lists that are part of other lists (sublists)
        3.2. Delete lists which contain 1 element with 1 usage

        """

        # Create semantic lists
        for hypersemant in self.tokens:
            self.sem_list = list()
            for token in self.tokens:
                # We add tokens by their similarity parameter with similar pos tag except the similar tokens
                if (hypersemant.similarity(token) > 0.6 and (hypersemant.similarity(token)) < 0.8 \
                        and hypersemant.lemma_ != token.lemma_ \
                        and hypersemant.pos_ == token.pos_ and token not in self.sem_list):
                    self.sem_list.append(token)
            self.sem_list = SemanticList(hypersemant, self.sem_list, self.tokens)
            if self.sem_list.number_of_unique_lemmas > 0 and self.sem_list not in self.sem_lists:
                self.sem_lists.append(self.sem_list)

        # An algorithm to delete sublists:
        # 1. Compare only 2+ elements lists.
        # 2. If an i-list is a sublist of a j-list and they are not the same lists we add +1 to a count variable.
        # 3. If after checking all j-lists the count is 0 --> the algorithm adds i-list to the temp_sem_lists.

        temp_sem_lists = []
        for i in range(len(self.sem_lists)):
            count = 0
            for j in range(len(self.sem_lists)):
                if self.sem_lists[i].lemmas.intersection(self.sem_lists[j].lemmas) == \
                self.sem_lists[i].lemmas and i != j:
                    count += 1
            if count == 0 and self.sem_lists[i].number_of_unique_lemmas > 0:
                temp_sem_lists.append(self.sem_lists[i])

        self.sem_lists = temp_sem_lists

        # Output
        LexDivCharacteristicCount = 0
        AverageSimilarityCount = 0
        LenCount = 0
        Count = 0
        for semantic_list in self.sem_lists:
            LexDivCharacteristicCount += semantic_list.lex_div_characteristic
            AverageSimilarityCount += semantic_list.avg_similarity
            LenCount += semantic_list.number_of_unique_lemmas
            Count += 1

        statistics_dict = {
            'LexDivCharacteristic average': LexDivCharacteristicCount / Count,
            'AverageSimilarity average': AverageSimilarityCount / Count,
            'Length average': LenCount / Count
        }

        return statistics_dict


class SemanticList:
    """
    This class is used to evaluate semantic relations by special property which is the relationship between the number
    of the unique lemmas of the list and the number of occurrences of those tokens in a content

    Arguments:
        hypersemant (spaCy token): A token by which similarity semantic list is constructed
        sem_list (list): A list contains semantically related tokens to the hypersemant
        nlp_doc (spaCy doc): Doc object which contains tokens from the essay

    Attributes:
        hypersemant (spaCy token): A token by which similarity semantic list is constructed
        list (list): A list contains semantically related tokens to the hypersemant
        lemmas (list): A list contains lemmas of the tokens
        number_of_unique_lemmas (int): The length of the list
        list_text (list): A list contains text of the tokens
        nlp_doc (spaCy doc): A doc object which contains tokens from the essay

    """

    def __init__(self, hypersemant, sem_list, tokens):
        self.hypersemant = hypersemant
        self.list = sem_list
        self.tokens = tokens

        # Delete duplicates
        unique_tokens = list()
        self.lemmas = list()
        for token in self.list:
            if token.lemma_ not in self.lemmas:
                unique_tokens.append(token)
                self.lemmas.append(token.lemma_)
        self.list = unique_tokens
        self.lemmas = set(self.lemmas)

        self.list_text = [token.text for token in unique_tokens]
        self.number_of_unique_lemmas = len(self.lemmas)

        # Calculate average similarity of the list by dividing sum of similarities by self.len
        if self.number_of_unique_lemmas > 0:
            self.avg_similarity = sum(
                [self.hypersemant.similarity(token) for token in self.list]) / self.number_of_unique_lemmas

        # Calculate the relationship between the unique lemmas of the list
        # and the number of uses of tokens in the text.
        # An algorithm is:
        # 1. Assume Counter of tokens' texts from a text to a count variable.
        # 2. Calculate number of occurrences for each token and add it to count_of_occurrences.
        # 3. Divide self.len by count_of_occurrences.
        count = Counter(tokens)
        count_of_occurrences = sum([count[token] for token in self.tokens])
        self.lex_div_characteristic = self.number_of_unique_lemmas / count_of_occurrences


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
    Selects synonyms with low frequency and range, synonyms from Academic Word List, with higher CEFR level
    to a given token.

    Methods:
    get_replacement_options
    """

    def __init__(self):
        pass

    def get_replacement_options(self):
        pass


class AnalysisOfVocabularyRichness:
    """

    """

    def __init__(self):
        pass


text = Text('wt2.txt')
print(len(text.get_tokens(True, True)))

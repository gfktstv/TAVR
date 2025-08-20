import spacy
from spacy.tokens import Doc
from spacy_ngram import NgramComponent

from nltk.corpus import wordnet as wn

from lexical_diversity import lex_div as ld

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats

import orjson

import itertools

from pathlib import Path

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("spacy-ngram")  # Pipeline for n-gram marking

# For the m-series macbooks
matplotlib.use("Agg")

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
        banned_pos = ["PROPN", "SYM", "PART", "CCONJ", "ADP", "X"]
        if include_content_words and include_functional_words:
            tokens = [token for token in self.nlp_doc]
            # Check for small text or if text not in English
            if len(tokens) < 50:
                raise TokensAreNotRecognized(
                    f"SpaCy module recognized only {len(tokens)} tokens which is less then 50"
                )
        elif include_content_words:
            tokens = [token for token in self.nlp_doc if (not token.is_stop) and (token.pos_ not in banned_pos)]
            # Check for small text or if text not in English
            if len(tokens) < 50:
                raise TokensAreNotRecognized(
                    f"SpaCy module recognized only {len(tokens)} tokens which is less then 50"
                )
        elif include_functional_words:
            tokens = [token for token in self.nlp_doc if token.is_stop or (token.pos_ in banned_pos)]

        if include_punct and (len(tokens) == 0):
            # For an option with only punctuational marks being returned
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

        # Clean from "\n\n" (they are needed in tokens for a web app to preserve paragraphs after marking up an essay)
        lemmas = [lemma for lemma in lemmas if lemma != "\n\n"]

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
    
    def get_n_grams(self):
        """
        Returns dict of 2-, 3-, 4-, 5-grams with following structure: key is type of n-gram (e.g. 2-gram), value is the list of such n-grams.
        Each n-gram is str in which tokens divided by 1 whitespace
        """
        tokens = self.get_tokens()
        n_grams = {
            "2-gram": list(), "3-gram": list(), "4-gram": list(), "5-gram": list()
        }
        for i in range(len(tokens) - 4):
            n_grams["2-gram"].append(f"{tokens[i]} {tokens[i + 1]}".lower())
            n_grams["2-gram"].append(f"{tokens[i]} {tokens[i + 2]}".lower())
            n_grams["3-gram"].append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}".lower())
            n_grams["4-gram"].append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]} {tokens[i + 3]}".lower())
            n_grams["5-gram"].append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]} {tokens[i + 3]} {tokens[i + 4]}".lower())
                
        n_grams["2-gram"].append(f"{tokens[len(tokens) - 3]} {tokens[len(tokens) - 4]}".lower())
        n_grams["2-gram"].append(f"{tokens[len(tokens) - 2]} {tokens[len(tokens) - 4]}".lower())
        n_grams["2-gram"].append(f"{tokens[len(tokens) - 2]} {tokens[len(tokens) - 3]}".lower())
        n_grams["3-gram"].append(f"{tokens[len(tokens) - 2]} {tokens[len(tokens) - 3]} {tokens[len(tokens) - 4]}".lower())
        n_grams["4-gram"].append(f"{tokens[len(tokens) - 1]} {tokens[len(tokens) - 2]} {tokens[len(tokens) - 3]} {tokens[len(tokens) - 4]}".lower())
        
        return n_grams


class _LexicalSophisticationMeasurements:
    """
    Class measures lexical sophistication of a given text by average word frequency & range, n-gram frequency & range,
    academic formulas frequency, content of academic vocabulary, level of vocabulary. In addition, it marks up tokens
    and n-grams with frequency, range and other characteristics.

    Note, that all tokens and n-grams will be marked up only after calling all methods or method get_full_data

    Arguments:
        text (_Text): instance of class _Text

    Attributes:
        marked_up_n_grams (dict of dict): Dictionary for n_grams with following structure: 
        key: n_gram, value: dictionary with freq 
        (for non-academic 2-,3-grams and for academic formulas), freq_bawe and freq_bnc 
        (for academic collocations), range (for non-academic 2-, 3-grams), 
        indicators a_formula and a_collocation and the length of the n-gram
        vocabulary_by_level_dict (dict): number of _tokens by CEFR level

    Methods:
        word_freq_range
        word_information
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
            self._n_grams = text.get_n_grams()
        # For replacement options
        elif token_list is not None:
            assert isinstance(token_list, list)
            assert all(isinstance(token, str) for token in token_list)
            assert isinstance(pos_tag, str)
            doc = Doc(nlp.vocab, words=token_list, pos=[pos_tag for _ in range(len(token_list))], lemmas=token_list)
            banned_pos = ["PROPN", "SYM", "PART", "CCONJ", "ADP", "X"]
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
                    "punct": False, "functional_word": True,
                    "concreteness": None, "familiarity": None, "imageability": None,
                    "age_of_acquisition": None, "meaningfulness_colorado": None, "meaningfulness_paivio": None,
                    "id": self._tokens.index(token)
                }
            elif token in self._content_tokens:
                self._marked_up_tokens[token] = {
                    "punct": False, "functional_word": False, "freq": 0, "range": 0,
                    "concreteness": None, "familiarity": None, "imageability": None,
                    "age_of_acquisition": None, "meaningfulness_colorado": None, "meaningfulness_paivio": None,
                    "academic": bool(), "level": None, "id": self._tokens.index(token)
                }
            else:
                self._marked_up_tokens[token] = {
                    "punct": True, "functional_word": False, "id": self._tokens.index(token)
                }
        
        if text is not None:
            # Dictionary for n_grams with following structure: 
            # key: n_gram
            # value: dictionary with freq (for non-academic 2-,3-grams and for academic formulas), 
            # freq_bawe and freq_bnc (for academic collocations), range (for non-academic 2-, 3-grams), 
            # indicators a_formula and a_collocation and the length of the n-gram
            self.marked_up_n_grams = dict()
            for n_gram in itertools.chain.from_iterable(self._n_grams.values()):
                self.marked_up_n_grams[n_gram] = {
                    "freq": int(), "freq_bawe": float(), "freq_bnc": float(), 
                    "range": int(), "a_formula": False, "a_collocation": False,
                    "len": len(n_gram.split())
                }

        # Lists consisting of bigrams and trigrams which are unrecognized in corpus. Used for n-gram proportion
        self._unrecognized_bigrams = list()
        self._unrecognized_trigrams = list()

        # Dictionary with keys as CEFR levels (A1, A2, B1, etc.) and appropriate number of _tokens in a text
        self.vocabulary_by_level_dict = dict()

        self._CURRENT_DIR = Path(__file__).resolve().parent
        self._DATA_IN_JSON_DIR = self._CURRENT_DIR / "corpora" / "data_in_json"

    def word_freq_range(self, for_replacement_options=False):
        """
        Add frequency and range of a token to self._marked_up_tokens.
        Calculates average word frequency & range for all words as well as for content and functional words separately.

        Returns: dict containing average word frequency & range for all words as well as for content and
        functional words separately.

        :param bool for_replacement_options: Only marks up tokens with CEFR level without return
        """
        # Load words frequency and range from corpus
        with open(self._DATA_IN_JSON_DIR / "brown_word_freq_range.json", "rb") as f:
            brown_word_freq_range = orjson.loads(f.read())

        # Create lists of frequencies and ranges for all words (AW), content words (CW) and functional words (FW)
        # to calculate average frequency and range
        aw_freqs, aw_ranges = list(), list()
        cw_freqs, cw_ranges = list(), list()
        fw_freqs, fw_ranges = list(), list()
        for token in self._tokens:
            try:
                word = f"{token.lemma_}_{token.pos_}".lower()
                word_freq, word_range = brown_word_freq_range[word].values()
                self._marked_up_tokens[token]["freq"] = word_freq
                self._marked_up_tokens[token]["range"] = word_range

                aw_freqs.append(word_freq)
                aw_ranges.append(word_range)
                if self._marked_up_tokens[token]["functional_word"]:
                    fw_freqs.append(word_freq)
                    fw_ranges.append(word_range)
                else:
                    cw_freqs.append(word_freq)
                    cw_ranges.append(word_range)
            except KeyError:
                # Ignores KeyError that occurs due to the absence of a token in the corpus
                continue

        if not for_replacement_options:
            measurements_dict = {
                "All words frequency": np.mean(aw_freqs),
                "Content words frequency": np.mean(cw_freqs),
                "Functional words frequency": np.mean(fw_freqs),
                "All words range": np.mean(aw_ranges),
                "Content words range": np.mean(cw_ranges),
                "Functional words range": np.mean(fw_ranges),
            }

            return measurements_dict

    def word_information(self):
        """
        Calculates average of word information scores such as familiarity, concreteness, imageability, meaningfulness,
        and age of acquisition using MRC psycholinguistic database. It should be noted that words which are not in
        MRC database are not considered in calculating average

        Returns: a dict of average familiarity, concreteness, imageability, meaningfulness by colorado and paivio norms,
        and age of acquisition
        """
        # Load mrc psycholinguistic database with required information
        with open(self._DATA_IN_JSON_DIR / "mrc_psycholinguistics.json", "rb") as f:
            mrc = orjson.loads(f.read())

        # Dictionary with pos tags for correct interpretation of spaCy pos tags in mrc database
        tags_dict = {
            "ADJ": "J", "ADP": "O",
            "ADV": "O", "AUX": "O",
            "CCONJ": "O", "DET": "O",
            "INTJ": "O", "NOUN": "N",
            "NUM": "O", "PART": "O",
            "PRON": "O", "SCONJ": "O",
            "VERB": "V", "X": "O",
            "PROPN": "O", "PUNCT": "O",
            "SPACE": "O", "SYM": "O",
        }
        # Mark up tokens with psycholinguistic information using dictionaries with word information
        for token, token_dict in self._marked_up_tokens.items():
            # Create key to MRC
            word = f"{token.lemma_}_{tags_dict[token.pos_]}".upper()

            if not token_dict["punct"]:
                for score in ["familiarity", "concreteness", "imageability", "age_of_acquisition",
                              "meaningfulness_colorado", "meaningfulness_paivio"]:
                    try:
                        self._marked_up_tokens[token][score] = mrc[word][score]
                    except KeyError:
                        # Ignore if token not in MRC
                        continue

        # List of scores
        scores = ["familiarity", "concreteness", "imageability", "age_of_acquisition",
                  "meaningfulness_colorado", "meaningfulness_paivio"]

        # Create empty dictionaries of word information scores
        aw_scores_dict = dict()
        cw_scores_dict = dict()
        fw_scores_dict = dict()
        for score in scores:
            aw_scores_dict[score] = list()
            cw_scores_dict[score] = list()
            fw_scores_dict[score] = list()

        # Fill dictionaries of word information scores
        for token, token_dict in self._marked_up_tokens.items():
            if not token_dict["punct"]:
                for score in scores:
                    if token_dict[score] is not None:
                        # Fill word information scores for all words
                        aw_scores_dict[score].append(token_dict[score])
                        # Fill word information for functional and content words
                        if token in self._content_tokens:
                            cw_scores_dict[score].append(token_dict[score])
                        elif token in self._functional_tokens:
                            fw_scores_dict[score].append(token_dict[score])

        # Fill mean score for each score type in intermediate_measurements_dict
        # and then merge it in measurements_dict
        measurements_dict = dict()
        for score in scores:
            intermediate_measurements_dict = {
                f"All words {score.replace("_", " ")}": np.mean(aw_scores_dict[score]),
                f"Content words {score.replace("_", " ")}": np.mean(cw_scores_dict[score]),
                f"Functional words {score.replace("_", " ")}": np.mean(fw_scores_dict[score]),
            }
            measurements_dict = measurements_dict | intermediate_measurements_dict

        return measurements_dict

    def n_gram_freq_range(self):
        """
        Create lists with bi-, trigrams and its frequency & range.
        Calculates average bi-, trigrams frequency & range.

        Returns: dict containing average frequency and range.

        """ 
        # Load n-grams frequency and range from corpus
        with open(self._DATA_IN_JSON_DIR / "brown_trigram_freq_range.json", "rb") as f:
            brown_trigram_freq_range = orjson.loads(f.read())
        with open(self._DATA_IN_JSON_DIR / "brown_bigram_freq_range.json", "rb") as f:
            brown_bigram_freq_range = orjson.loads(f.read())
            
        # Create lists of frequencies and ranges for calculating average values further
        bigram_frequencies, bigram_ranges = list(), list()
        trigram_frequencies, trigram_ranges = list(), list()
        
        for bigram in self._n_grams["2-gram"]:
                try:
                    bigram_freq, bigram_range = brown_bigram_freq_range["__".join(bigram.split())].values()

                    bigram_frequencies.append(bigram_freq)
                    bigram_ranges.append(bigram_range)

                    # Add bigram marked with frequency and range to a dict
                    self.marked_up_n_grams[bigram] = {
                    "freq": bigram_freq, "freq_bawe": int(), "freq_bnc": int(), 
                    "range": bigram_range, "a_formula": False, "a_collocation": False,
                    "len": 2
                    }
                except KeyError:
                    # Unrecognized bigrams for bigram proportion
                    self._unrecognized_bigrams.append(bigram)
                    continue
            
        for trigram in self._n_grams["3-gram"]:
            try:
                trigram_freq, trigram_range = brown_trigram_freq_range["__".join(trigram.split())].values()

                trigram_frequencies.append(trigram_freq)
                trigram_ranges.append(trigram_range)

                # Add trigram marked with frequency and range to a dict
                self.marked_up_n_grams[trigram] = {
                    "freq": trigram_freq, "freq_bawe": int(), "freq_bnc": int(), 
                    "range": trigram_range, "a_formula": False, "a_collocation": False,
                    "len": 3
                    }
            except KeyError:
                # Unrecognized trigrams for trigram proportion
                self._unrecognized_trigrams.append(trigram)
                continue
        
        measurements_dict = {
            "Bigram frequency": np.mean(bigram_frequencies),
            "Bigram range": np.mean(bigram_ranges),
            "Trigram frequency": np.mean(trigram_frequencies),
            "Trigram range": np.mean(trigram_ranges)
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
            "Bigram proportion": len(self._unrecognized_bigrams) / len(self._n_grams["2-gram"]),
            "Trigram proportion": len(self._unrecognized_trigrams) / len(self._n_grams["3-gram"])
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
        trigrams = list([trigram for trigram in self._n_grams["3-gram"] if trigram in self.marked_up_n_grams.keys()])
        bigrams = list([bigram for bigram in self._n_grams["2-gram"] if bigram in self.marked_up_n_grams.keys()])

        # Load lists of bigrams and trigrams from a corpus
        with open(self._DATA_IN_JSON_DIR / "brown_bigrams.json", "rb") as f:
            brown_bigrams = orjson.loads(f.read())
        with open(self._DATA_IN_JSON_DIR / "brown_trigrams.json", "rb") as f:
            brown_trigrams = orjson.loads(f.read())

        # Create lists of bigrams and trigrams normalized frequency from an essay and from a corpus
        bigrams_normalized_frequency_essay = list()
        bigrams_normalized_frequency_corpus = list()
        trigrams_normalized_frequency_essay = list()
        trigrams_normalized_frequency_corpus = list()
        for n_gram, n_gram_dict in self.marked_up_n_grams.items():
            if n_gram_dict["len"] == 2:
                normalized_frequency = bigrams.count(n_gram) / len(bigrams)
                bigrams_normalized_frequency_essay.append(normalized_frequency)

                normalized_frequency = n_gram_dict["freq"] / len(brown_bigrams)
                bigrams_normalized_frequency_corpus.append(normalized_frequency)
            elif n_gram_dict["len"] == 3:
                normalized_frequency = trigrams.count(n_gram) / len(trigrams)
                trigrams_normalized_frequency_essay.append(normalized_frequency)

                normalized_frequency = n_gram_dict["freq"] / len(brown_trigrams)
                trigrams_normalized_frequency_corpus.append(normalized_frequency)
            else:
                continue
        
        # Calculate correlation (accuracy)
        bigram_accuracy, bigram_p = stats.pearsonr(
            bigrams_normalized_frequency_corpus, bigrams_normalized_frequency_essay
        )
        trigram_accuracy, trigram_p = stats.pearsonr(
            trigrams_normalized_frequency_corpus, trigrams_normalized_frequency_essay
        )

        measurements_dict = {
            "Bigram accuracy": .0 if np.isnan(bigram_accuracy) else bigram_accuracy,
            "Trigram accuracy": .0 if np.isnan(trigram_accuracy) else trigram_accuracy
        }

        return measurements_dict

    def academic_n_grams(self):
        """
        Marks up n-grams if they are in the Academic Formulas List or in the Academic Collocations Lust. 
        Returns number of such n-grams in the text and academic collocations frequencies
        """
        # Load academic formulas list
        with open(self._DATA_IN_JSON_DIR / "afl.json", "rb") as f:
            academic_formulas_list = dict(orjson.loads(f.read()))
        # Load academic collocations list
        with open(self._DATA_IN_JSON_DIR / "acl.json", "rb") as f:
            academic_collocations_list = dict(orjson.loads(f.read()))
            
        # Create lists to calculate mean frequency (BNC and BAWE) for collocations 
        collocations_frequencies_bnc, collocations_frequencies_bawe = list(), list()
            
        count = 0
        for n_gram in self.marked_up_n_grams.keys():
            if n_gram in academic_formulas_list.keys():
                count += 1
                self.marked_up_n_grams[n_gram] = {
                    "freq": int(academic_formulas_list[n_gram]), "freq_bawe": int(), "freq_bnc": int(), 
                    "range": int(), "a_formula": True, "a_collocation": False,
                    "len": len(n_gram.split())
                    }
                self.marked_up_n_grams[n_gram]["a_formula"] = True
            elif n_gram in academic_collocations_list.keys():
                count += 1
                
                freq_bawe = float(academic_collocations_list[n_gram][0])
                freq_bnc = float(academic_collocations_list[n_gram][1])
                
                collocations_frequencies_bawe.append(freq_bawe)
                collocations_frequencies_bnc.append(freq_bnc)
                
                self.marked_up_n_grams[n_gram] = {
                    "freq": int(), "freq_bawe": freq_bawe, 
                    "freq_bnc": freq_bnc, 
                    "range": int(), "a_formula": False, "a_collocation": True,
                    "len": len(n_gram.split())
                    }
            else:
                continue
            
        measurements_dict = {
            "Number of academic n-grams": count, 
            "Academic collocations frequency BAWE": np.mean(collocations_frequencies_bawe) if len(collocations_frequencies_bawe) > 0 else .0,
            "Academic collocations frequency BNC": np.mean(collocations_frequencies_bnc) if len(collocations_frequencies_bnc) > 0 else .0,
        }
            
        return measurements_dict

    def academic_vocabulary(self):
        """
        Marks up tokens if they are in the New Academic Word List. 
        Returns number and percentage of such tokens in the text
        """
        # Load new academic word list in json
        with open(self._DATA_IN_JSON_DIR / "nawl.json", "rb") as f:
            new_academic_word_list = orjson.loads(f.read())

        count = 0
        for token in self._content_tokens:
            if token.lex.text in new_academic_word_list:
                count += 1
                self._marked_up_tokens[token]["academic"] = True

        measurements_dict = {
            "Number of academic words": count, 
            "Percentage of academic words": count / len(self._content_tokens)
        }

        return measurements_dict

    def vocabulary_by_level(self, for_replacement_options=False):
        """
        Marks up tokens with CEFR level (A1, A2, B1, etc.) and calculates number of tokens by each CEFR level.

        Returns a dictionary.

        :param bool for_replacement_options: Only marks up tokens with CEFR level without return
        """
        # Load EFLLex corpus
        with open(self._DATA_IN_JSON_DIR / "tokens_with_CEFR_level_efllex.json", "rb") as f:
            tokens_with_CEFR_level_corpus = orjson.loads(f.read())

        tokens_by_CEFR_level = {
            "A1": list(), "A2": list(),
            "B1": list(), "B2": list(),
            "C1": list(), "C2": list()
        }
        # Dict to convert spaCy"s tags to the corpus" tags. Some of the tags are empty (None) because they are
        # insignificant or missing in the corpus"s tags
        tags_dict = {
            "ADJ": "JJ", "ADP": None,
            "ADV": "RB", "AUX": "VB",
            "CCONJ": None, "DET": None,
            "INTJ": None, "NOUN": "NN",
            "NUM": "CD", "PART": None,
            "PRON": [" NN", "EX", "PRP", "WP"], "SCONJ": "IN",
            "VERB": ["MD", "VB"], "X": "XX",
            "PROPN": None, "PUNCT": None,
            "SPACE": None, "SYM": None,
        }

        for token in self._content_tokens:
            if type(tags_dict[token.pos_]) is str:
                # Some words may not be in corpus, therefore we will use try/except
                try:
                    level = tokens_with_CEFR_level_corpus[f"{token.lemma_}_{tags_dict[token.pos_]}"]
                    tokens_by_CEFR_level[level].append(token)
                    self._marked_up_tokens[token]["level"] = level
                except KeyError:
                    self._marked_up_tokens[token]["level"] = "C2"
                    tokens_by_CEFR_level["C2"].append(token)
            # If there are several tags for one spaCy tag we try all of them
            elif type(tags_dict[token.pos_]) is list:
                for tag_option in tags_dict[token.pos_]:
                    try:
                        level = tokens_with_CEFR_level_corpus[f"{token.lemma_}_{tag_option}"]
                        tokens_by_CEFR_level[level].append(token)
                        self._marked_up_tokens[token]["level"] = level
                    except KeyError:
                        pass
                if self._marked_up_tokens[token]["level"] is None:
                    self._marked_up_tokens[token]["level"] = "C2"
                    tokens_by_CEFR_level["C2"].append(token)
            elif tags_dict[token.pos_] is None:
                continue
            else:
                print(f"Token {token.text} has unknown part of speech tag that is {token.pos_}")

        self.vocabulary_by_level_dict = {
            "A1 words": len(tokens_by_CEFR_level["A1"]), "A2 words": len(tokens_by_CEFR_level["A2"]),
            "B1 words": len(tokens_by_CEFR_level["B1"]), "B2 words": len(tokens_by_CEFR_level["B2"]),
            "C1 words": len(tokens_by_CEFR_level["C1"]), "C2 words": len(tokens_by_CEFR_level["C2"]),
        }

        level_weight = {
            "A1": 9.8245136, "A2": 15.03967718, "B1": 44.12832192,
            "B2": -25.11806657, "C1": 50.84860667, "C2": -27.24657368
        }

        vocabulary_metric = list()
        # for level in tokens_by_CEFR_level.keys():
        #     for i in range(self.vocabulary_by_level_dict[f"{level} words"]):
        #         vocabulary_metric.append(i * level_weight[level])
        for level in level_weight.keys():
            vocabulary_metric.append(self.vocabulary_by_level_dict[f"{level} words"] * level_weight[level])

        if not for_replacement_options:
            measurements_dict = {
                "Vocabulary": np.sum(vocabulary_metric)
            }

            return measurements_dict

    def get_full_data(self):
        """
        Combines data from all methods into one dictionary.

        Returns: dictionary with data from all methods
        """
        word_freq_range_data = self.word_freq_range()
        word_information_data = self.word_information()
        n_gram_freq_range_data = self.n_gram_freq_range()
        n_gram_proportion_data = self.n_gram_proportion()
        n_gram_accuracy_data = self.n_gram_accuracy()
        academic_vocabulary_data = self.academic_vocabulary()
        academic_n_grams_data = self.academic_n_grams()
        vocabulary_by_level_data = self.vocabulary_by_level()

        full_data = (word_freq_range_data | word_information_data | n_gram_freq_range_data | n_gram_proportion_data |
                     n_gram_accuracy_data | academic_vocabulary_data | academic_n_grams_data | vocabulary_by_level_data)

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
                    if (value["functional_word"] is False) and (value["punct"] is False)}


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
            "TTR": ld.ttr(self._lemmatized_text),
            "Root TTR": ld.root_ttr(self._lemmatized_text),
            "Log TTR": ld.log_ttr(self._lemmatized_text),
            "Maas TTR": ld.maas_ttr(self._lemmatized_text),
            "D": ld.hdd(self._lemmatized_text),
            "MTLD": ld.mtld(self._lemmatized_text),
            "MTLD MA Wrap": ld.mtld_ma_wrap(self._lemmatized_text)
        }
        return data_dict


class TokensAreNotRecognized(Exception):
    pass


class TextAnalysis:
    """
    Analysis of a given essay by lexical diversity and lexical sophistication.
    Provides tables with trigrams with the biggest frequency/range, lexical diversity indices data, academic formulas, 
    a pie chart with the CEFR levels (A1, A2, B1, etc.), and appropriate number of words from a given essay.

    :param str essay: Essay presented as string
    """

    def __init__(self, essay: str):
        self._text = _Text(essay)
        self._lex_div = _LexicalDiversityMeasurements(self._text)
        self._lex_sop = _LexicalSophisticationMeasurements(self._text)

        self._lexical_sophistication_measurements = self._lex_sop.get_full_data()
        self._lexical_diversity_measurements = self._lex_div.indices_data()
        
        self._marked_up_tokens = self._lex_sop.get_marked_up_tokens()
        self._marked_up_tokens_wout_functional = self._lex_sop.get_marked_up_tokens(False)
            
        self._marked_up_n_grams = self._lex_sop.marked_up_n_grams
        self._vocabulary_by_level_dict = self._lex_sop.vocabulary_by_level_dict
        
    @staticmethod
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{p:.1f}%\n({v:d})".format(p=pct, v=val)

        return my_autopct
        
    def create_vocabulary_chart(self, small: bool = False):
        """
        Creates vocabulary chart in tmp/
        """
        _, ax = plt.subplots(facecolor=(0.1, 0.2, 0.5, 0))
        
        vocabulary_level_count = list(self._vocabulary_by_level_dict.values())
        levels_of_vocabulary = ["A1", "A2", "B1", "B2", "C1", "C2"]
        level_colors = ["#FFE89C", "#FFCF32", "#66C4D8",
                        "#5282F2", "#9C99FF", "#6B66FF"]
        
        TMP_DIR = Path(__file__).resolve().parent / "tmp"
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        
        if small:
            ax.pie(vocabulary_level_count,
                labels=levels_of_vocabulary,
                autopct="%1.1f%%",
                colors=level_colors
                )
            path = TMP_DIR / "vocabulary_chart_small.png"
        else:
            ax.pie(vocabulary_level_count,
               labels=levels_of_vocabulary,
               autopct=self.make_autopct(vocabulary_level_count),
               colors=level_colors,
               explode=(0.1, 0.1, 0, 0, 0, 0),
               textprops={"fontsize": 9}
               )
            path = TMP_DIR / "vocabulary_chart.png"
            
        plt.savefig(path,
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=500.0)

    def get_trigrams_dataframe(self, entries_limit: int | None = None):
        """
        Creates a pandas DataFrame with trigrams with the biggest frequency or range from a given essay.
        
        :param int | None entries_limit: Number of rows to show

        :return: A pandas DataFrame
        """
        sorted_n_grams = sorted(self._marked_up_n_grams.items(), 
                                key=lambda x: x[1]["freq"], reverse=True)
        # Dictionary that will be converted into CSV table
        trigrams = {
            "Trigram": list(), "Frequency": list(), "Range": list()
        }
        for n_gram_tuple in sorted_n_grams:
            if n_gram_tuple[1]["len"] == 3:
                trigrams["Trigram"].append(n_gram_tuple[0])
                trigrams["Frequency"].append(n_gram_tuple[1]["freq"])
                trigrams["Range"].append(n_gram_tuple[1]["range"])
        trigrams = pd.DataFrame(trigrams)
        trigrams.fillna("-", inplace=True)
        
        if entries_limit is not None:
            return trigrams.iloc[:entries_limit, :]
        
        return trigrams

    def get_academic_formulas_dataframe(self, entries_limit: int | None = None):
        """
        Creates a pandas DataFrame with academic formulas. 
        
        :param int | None entries_limit: Number of rows to show
        
        :return: A pandas DataFrame
        """
        # Dictionary that will be converted into CSV table
        academic_formulas = {
            "Academic formula": list()
        }
        for n_gram, n_gram_dict in self._marked_up_n_grams.items():
            if n_gram_dict["a_formula"]:
                academic_formulas["Academic formula"].append(n_gram)
        academic_formulas = pd.DataFrame(academic_formulas)

        if academic_formulas.empty:
            academic_formulas = pd.DataFrame({"Academic formula": ["Not found"]})
            
        if entries_limit is not None:
            return academic_formulas.iloc[:entries_limit, :]

        return academic_formulas
    
    def get_academic_collocations_dataframe(self, entries_limit: int | None = None):
        """
        Creates a pandas DataFrame with academic collocations. 
        
        :param int | None entries_limit: Number of rows to show
        
        :return: A pandas DataFrame
        """
        # Dictionary that will be converted into CSV table
        academic_collocations = {
            "Academic collocation": list()
        }
        for n_gram, n_gram_dict in self._marked_up_n_grams.items():
            if n_gram_dict["a_collocation"]:
                academic_collocations["Academic collocation"].append(n_gram)
        academic_collocations = pd.DataFrame(academic_collocations)

        if academic_collocations.empty:
            academic_collocations = pd.DataFrame({"Academic collocation": ["Not found"]})
            
        if entries_limit is not None:
            return academic_collocations.iloc[:entries_limit, :]

        return academic_collocations

    def get_academic_words_dataframe(self, entries_limit: int | None = None):
        """
        Creates a pandas DataFrame with academic words. 
        
        :param int | None entries_limit: Number of rows to show
        
        :return: A pandas DataFrame
        """
        academic_words = {
            "Academic word": list()
        }

        for token, token_dict in self._marked_up_tokens_wout_functional.items():
            if token_dict["academic"]:
                academic_words["Academic word"].append(token)
        academic_words = pd.DataFrame(academic_words)

        if academic_words.empty:
            academic_words = pd.DataFrame({"Academic word": ["Not found"]})
            
        if entries_limit is not None:
            return academic_words.iloc[:entries_limit, :]

        return academic_words

    def get_stats_dataframe(self):
        """
        Creates a csv table with data from indices and percentage of academic words.

        :return: A pandas DataFrame
        """
        stats_dict = {"metric": list(), "value": list()}
        stats_dict["metric"].append("TTR")
        stats_dict["value"].append(round(self._lexical_diversity_measurements["TTR"], 2))
        stats_dict["metric"].append("Academic words")
        stats_dict["value"].append(str(self._lexical_sophistication_measurements["Number of academic words"]))
        stats_dict["metric"].append("Average trigram frequency")
        stats_dict["value"].append(round(self._lexical_sophistication_measurements["Trigram frequency"], 2))
        stats = pd.DataFrame(stats_dict)
        
        return stats

    def get_recurring_lemmas_dataframe(self, include_functional_words: bool = False, entries_limit: int | None = None):
        """
        Returns a pandas DataFrame of lemmas which occur in a text 2 or more times

        :param bool include_functional_words: Include stopwords, proper nouns, symbols, particles, adposition,
        coordinating conjunction and unknown parts of speech
        """
        recurring_lemmas = {
            "Lemma": list(), "Occurrences": list()
        }
        if include_functional_words:
            lemmas_occurrences = self._text.get_lemmas_occurrences(include_functional_words=True)
        else:
            lemmas_occurrences = self._text.get_lemmas_occurrences(include_functional_words=False)

        for lemma, occurrences in lemmas_occurrences.items():
            if occurrences >= 2:
                recurring_lemmas["Lemma"].append(lemma)
                recurring_lemmas["Occurrences"].append(occurrences)
        recurring_lemmas = pd.DataFrame(recurring_lemmas).sort_values(by="Occurrences", ascending=False)
        
        if entries_limit is not None:
            return recurring_lemmas.iloc[:entries_limit, :]

        return recurring_lemmas

    def get_level(self):
        """Returns a CEFR level (A1, A2, B1, etc.) based on TTR value as the metric with the biggest correlation"""
        TTR = self._lexical_diversity_measurements["TTR"]
        coefficients = [-59.4178015, 69.99997391, -13.2177359]

        degree = len(coefficients) - 1
        score = sum([coefficients[i] * (TTR ** (degree - i)) for i in range(len(coefficients))])
        if score <= 4:
            return f"A2"
        elif score <= 5:
            return f"B1"
        elif score <= 6.5:
            return f"B2"
        elif score <= 8:
            return f"C1"
        else:
            return f"C2"

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


def main():
    pass


if __name__ == "__main__":
    main()

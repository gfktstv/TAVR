import en_core_web_lg

from collections import Counter

nlp = en_core_web_lg.load()


class Text:
    """
    Class to work with a whole text of the essay

    Arguments:
        source (str): Source of Data consisting an essay

    Attributes:
        content (str): Content of the essay
        nlp_doc (spaCy doc): Doc object which contains tokens from the essay
        tokens (list): All tokens from the text (without punctuation's marks)
        tokens_without_stop (list): Tokens variable without stop words
        unique_tokens_texts (list): Tokens without token.text duplicates
        unique_tokens_lemmas (list): Tokens without token.lemma duplicates

    Methods:
        mistakes_correction
        mistakes_statistics

    """

    def __init__(self, source):
        self.source = source
        with open(f'{self.source}', 'r') as file:
            self.content = file.read().replace('\n', ' ')

        self.nlp_doc = nlp(self.content)
        self.tokens = [token for token in self.nlp_doc if not token.is_punct]
        self.tokens_without_stop = [token for token in self.nlp_doc if not token.is_stop]

        self.unique_tokens_texts = []
        unique_tokens_text = []
        for token in self.tokens:
            if token.text not in unique_tokens_text:
                self.unique_tokens_texts.append(token)
                unique_tokens_text.append(token.text)

        self.unique_tokens_lemmas = []
        unique_tokens_lemmas = []
        for token in self.tokens:
            if token.lemma_ not in unique_tokens_lemmas:
                self.unique_tokens_lemmas.append(token)
                unique_tokens_lemmas.append(token.lemma_)

    def __str__(self):
        return text.content


class CorpusDrivenMethods:
    """
    Class which evaluates vocabulary richness of the text by using corpus driven methods.
    It contains several methods such as word frequency, N-gram statistics, word range,
    word specificity, vocabulary CEFR level and academic vocabulary percentage.
    All of them return statistics organized by dictionaries. At the end, statistics
    method returns concatenated statistics' dictionaries from the methods.

    Attributes:

    Methods:
        word_frequency_statistics
        N_gram_statistics
        word_range_statistics
        word_specificity_statistics
        vocabulary_by_CEFR_level_statistics
        academic_vocabulary_statistics
        statistics

    """

    def __init__(self):
        pass


class MTLD:
    """
    Class MTLD evaluates lexical diversity (part of vocabulary richness) using MTLD index.
    Returns a dictionary with statistics

    Attributes:

    Methods:
        statistics

    """

    def __init__(self):
        pass


class SemanticRelations:
    """
    Class which evaluates vocabulary richness of the text on semantic relations.
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
        self.tokens = text_class_variable.unique_tokens_texts
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
            if self.sem_list.len > 0 and self.sem_list not in self.sem_lists:
                self.sem_lists.append(self.sem_list)

        # An algorithm to delete sublists:
        # 1. Compare only 2+ elements lists
        # 2. If an i-list is a sublist of a j-list and they are not the same lists we add +1 to a count variable
        # 3. If after checking all j-lists the count is 0 --> the algorithm adds i-list to the temp_sem_lists

        temp_sem_lists = []
        for i in range(len(self.sem_lists)):
            count = 0
            for j in range(len(self.sem_lists)):
                if self.sem_lists[i].lemmas.intersection(self.sem_lists[j].lemmas) == self.sem_lists[i].lemmas and i!=j:
                    count += 1
            if count == 0 and self.sem_lists[i].len > 0:
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
            LenCount += semantic_list.len
            Count += 1

        statistics_dict = {
            'LexDivCharacteristic average': LexDivCharacteristicCount / Count,
            'AverageSimilarity average': AverageSimilarityCount / Count,
            'Length average': LenCount / Count
        }

        return statistics_dict


class SemanticList:
    """
    This class is used to evaluate semantic relations by special property which is the relationship between
    the unique lemmas of the list and the number of uses of tokens in the text

    Arguments:
        hypersemant (spaCy token): A token by which similarity semantic list is constructed
        sem_list (list): A list contains semantically related tokens to the hypersemant
        nlp_doc (spaCy doc): Doc object which contains tokens from the essay

    Attributes:
        hypersemant (spaCy token): A token by which similarity semantic list is constructed
        list (list): A list contains semantically related tokens to the hypersemant
        lemmas (list): A list contains lemmas of the tokens
        len (int): The length of the list
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
        self.len = len(self.lemmas)

        # Calculate average similarity of the list by dividing sum of similarities by self.len
        if self.len > 0:
            self.avg_similarity = sum([self.hypersemant.similarity(token) for token in self.list]) / self.len

        # Calculate the relationship between the unique lemmas of the list
        # and the number of uses of tokens in the text.
        # An algorithm is:
        # 1. Assume Counter of tokens' texts from a text to a count variable.
        # 2. Calculate number of occurrences for each token and add it to count_of_occurrences.
        # 3. Divide self.len by count_of_occurrences.
        count = Counter(tokens)
        count_of_occurrences = sum([count[token] for token in self.tokens])
        self.lex_div_characteristic = self.len / count_of_occurrences


text = Text('wt2.txt')
test = SemanticRelations(text)
print(test.statistics())
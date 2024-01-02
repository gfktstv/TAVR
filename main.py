import spacy
import en_core_web_lg

from enchant.checker import SpellChecker

from collections import Counter

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
        tokens (set): Valuable tokens from the content (without puncts and stop words)
        sem_list (list): Paradigm of semantic related words to the one
        sem_lists (list): List of semantic lists

    Methods:
        get_statistics_of_semantic_relations
    """

    def __init__(self, text_class_variable):
        super().__init__(text_class_variable)
        self.content = text_class_variable.content
        self.tokens = list()
        self.sem_list = list()
        self.sem_lists = list()

    def get_semantic_relations_stats(self):
        """
        This algorithm extracts from text semantic lists - paradigm of words which semantically connected to the one.

        Description of the algorithm:
        1. Extract valuable tokens from the content.
        2. For each token search semantically related words in the text (>0.65 similarity, but <0.8)
        3. Add each semantic list to a list of semantic lists
        4. Clear semantic lists:
        4.1. Delete lists that are part of other lists (sublists)
        4.2. Delete lists which contain 1 element with 1 usage
        """
        self.tokens = [
            token for token in self.nlp_doc
            if not token.is_stop and not token.is_punct
        ]

        # Delete duplicates
        unique_tokens_text = []
        unique_tokens = []
        for token in self.tokens:
            if token.text not in unique_tokens_text:
                unique_tokens.append(token)
                unique_tokens_text.append(token.text)

        self.tokens = unique_tokens

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
            if count == 0 and self.sem_lists[i].len > 2:
                temp_sem_lists.append(self.sem_lists[i])

        self.sem_lists = temp_sem_lists

        return self.sem_lists


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
    def __init__(self, hypersemant, sem_list, nlp_doc):
        self.hypersemant = hypersemant
        self.list = sem_list
        self.nlp_doc = nlp_doc

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
        tokens = [
            token.text for token in self.nlp_doc
            if not token.is_stop and not token.is_punct
        ]
        count = Counter(tokens)
        count_of_occurrences = sum([count[token] for token in tokens])
        self.lex_div_characteristic = self.len / count_of_occurrences


text = Text('wt2.txt')
test = SemanticRelationsStats(text)
for semantic_list in test.get_semantic_relations_stats():
    print(semantic_list.list, semantic_list.lex_div_characteristic, semantic_list.avg_similarity)
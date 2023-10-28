import re
import nltk
from nltk.corpus import stopwords

with open('wt2.txt', 'r') as file:
    text1 = file.read().replace('\n', '')


class WT2:
    def __init__(self, content):
        self.content = content.lower()

    def lr_evaluator(self):
        #
        # 1. Deleting punctuation marks
        #
        dict_of_punctuation_marks = {
            '!': '', '.': '',  # ! and .
            ',': '', ';': '',  # , and ;
            '...': '', ':': '',  # ... and :
            '(': '', ')': '',  # ()
            '"': '', "'": '',  # ' and "
            '-': '', '—': '',  # - and —
            '?': '', '_': '',  # ? and _
        }

        # Create a pattern (regular expression) from the dictionary keys
        pattern_from_dict = re.compile(
            "(%s)" % "|".join(map(re.escape, dict_of_punctuation_marks.keys()))
        )

        # For each match, look-up corresponding value in dictionary
        self.content = pattern_from_dict.sub(
            lambda m: dict_of_punctuation_marks[m.group()], self.content
        )
        #
        # 2. Tokenization
        #
        self.content = nltk.word_tokenize(self.content)
        #
        # 3. Deleting stopwords
        #
        unique_words = []
        for i in range(len(self.content)):
            if self.content[i] not in stopwords.words('english'):
                unique_words.append(self.content[i])
        #
        # 4. Deleting unnecessary parts of speech
        #
        unique_words_pos = nltk.pos_tag(unique_words)

        necessary_pos_list = [
            'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VBZ',
            'VBP', 'VBN', 'VBD', 'VBG', 'VB', 'RBS', 'RB', 'RBR'
        ]
        unique_words = []
        for i in range(len(unique_words_pos)):
            if unique_words_pos[i][1] in necessary_pos_list:
                unique_words.append(unique_words_pos[i][0])


text_WT2 = WT2(text1)
print(text_WT2.lr_evaluator())

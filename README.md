# the Tool for Analysis of Vocabulary Richness
## Description
TAVR is a python package and web app for analyzing of vocabulary richness by lexical diversity indexes (TTR, MTLD, MTLD-W) and lexical sophistication methods (word frequency, range, CEFR level; n-gram frequency, range; academic words and formulas)
## How to use python package?
First off, clone the repository and import tavr
```
from tavr import *
```
Now you will be able to analyze text by TextAnalysis class and get replacement options to spaCy token by TokenReplacementOptions class, let's consider them in order
### Text analysis
Initialize your text in str format, it's better to limit your text under 2500 characters (350 words)
```
tavr = TextAnalysis(essay)
```
Get recurring lemmas from a given text in pandas DataFrame
```
recurring_lemmas_df = tavr.get_recurring_lemmas_dataframe(including_insignificant=False)
```
Get most frequent trigrams from a given text in pandas DataFrame
```
trigrams_df = tavr.get_trigrams_dataframe()
```
Get academic formulas (phrases) from a given text in pandas DataFrame
```
academic_formulas_df = tavr.get_academic_formulas_dataframe()
```
Get TTR value, number of academic words and average trigram frequency organized in pandas DataFrame
```
stats_df = tavr.get_stats_dataframe()
```
Get marked up tokens with frequency, range and other characteristics in dict format from a given text
```
marked_up_tokens_dict = tavr.marked_up_tokens
```
Get marked up n-grams with frequency, range and other characteristics in dict format from a given text
```
marked_up_n_grams_dict = tavr.marked_up_n-grams
```
Moreover, you can get access to all lexical diversity and lexical sophistication measurements by these attributes
```
lexical_diversity = tavr.lexical_diversity_measurements
lexical_sophistication = tavr.lexical_sophistication_measurements
```
Documentation for all classes and methods provided and you can see it in your IDE on mouse over method/class (TAVR is a pycharm project, therefore it will be better to use pycharm)
### Replacement options
Initialize marked up tokens from a given text (you can get it from TextAnalysis class)
```
tavr = TextAnalysis(essay)  # essay should be in str format
marked_up_tokens_dict = tavr.marked_up_tokens
replacements_options = TokenReplacementOptions(marked_up_tokens_dict)
```
And get replacements in list format
```
replacements = replacements_options.get_replacement_options(token, return_token_text=False)
```
## How to use web app?
### Launching
To use web app you should launch file connect_to_tavr.py and than tavr_main.html.
Notice that requests may take time to get from web to python so wait until your essay will be analyzed (10-15 seconds). The same for replacements available by clicking on word after analyzing.
### Instructions
1. Enter your essay. You must enter at least 1500 characters and no more than 2500 characters. You can see number of characters in right down angle.
2. Click on button "Analyze" and wait for 10-15 seconds.
3. Now you can see chart, tables and you level below marked up essay. You can see hints on mouse over them.
4. By clicking on words that are higlighted with a colour you can get replacements (small window). By clicking on replacements the word in the text will be replaced with replacement.
5. Finally you can copy your text by clicking on button "Copy" (replacements that you did will be included) or clear it all by clicking on button "Clear". 

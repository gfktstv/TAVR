from tavr import _LexicalSophisticationMeasurements

from nltk.corpus import wordnet as wn

_wn_pos_tag_map = {
        "N": wn.NOUN, "V": wn.VERB, "J": wn.ADJ, "R": wn.ADV, "A": wn.ADJ
}

def get_replacement_options(token: str, pos_tag: str, limit: int = 3):
    """
    :param str token: Token for which replacement options are needed
    :param str pos_tag: Part of speech tag from Spacy lib
    :param int limit: Limit the number of replacement options. 
    """
    wn_post_tag = _wn_pos_tag_map[pos_tag[0]]
    
    synonyms = list()
    for synset in wn.synsets(token, pos=wn_post_tag):
        for synonym in synset.lemmas():
            syn_name = synonym.name()
            if syn_name not in synonyms:
                synonyms.append(syn_name)
    
    words_to_mark_up = [token] + synonyms
    lexical_sophistication = _LexicalSophisticationMeasurements(
                                 token_list=words_to_mark_up, 
                                 pos_tag=pos_tag)
    
    lexical_sophistication.word_freq_range(for_replacement_options=True)
    lexical_sophistication.vocabulary_by_level(for_replacement_options=True)
    
    marked_up_tokens = lexical_sophistication.get_marked_up_tokens(
        include_functional_words=False)
    
    token_dict = list(marked_up_tokens.items())[0][1]
    curr_level = token_dict["level"]
    curr_freq = token_dict["freq"]
    curr_range = token_dict["range"]
    
    marked_up_tokens_wout_unknown = {t: t_dict for t, t_dict
                                     in list(marked_up_tokens.items())[1:]
                                     if (t_dict["level"] != "C2")
                                     and (t_dict["freq"] != 0)
                                     and (t_dict["range"] != 0)}
    
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    replacement_options = {
        t: t_dict for t, t_dict 
        in marked_up_tokens_wout_unknown.items()
        if (t_dict["freq"] < curr_freq) 
        or (t_dict["range"] < curr_range)
        or (levels.index(t_dict["level"]) > levels.index(curr_level))
    }
    replacement_options = sorted(
        replacement_options.items(), 
        key=lambda token_tuple: token_tuple[1]["level"],
        reverse=True
    )
    replacement_options_levels = [t_dict["level"] for _, t_dict 
                                  in replacement_options]
    
    replacement_options = [t.text.replace("_", " ") 
                           for t, _ in replacement_options]
    
    return replacement_options[:limit], replacement_options_levels
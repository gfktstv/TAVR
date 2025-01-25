import orjson
import json

from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_cors import CORS

import tavr


class Connection:
    def __init__(self):
        self._app = Flask(__name__)
        CORS(self._app)
        self._app.app_context().push()
        self.tokens_text = list()
        self.marked_up_tokens = dict()
        self.token_id_dict = dict()

    def connect(self):
        @self._app.route('/get_tables', methods=['POST'])
        def get_tables():
            """Receives a data from the web page and returns tables from TAVR TextAnalysis"""
            # Get the JSON data from the request and extracts input data (essay)
            data = request.get_json()
            essay = data.get('data')

            tavr_text_analysis = tavr.TextAnalysis(essay)

            trigrams, stats, academic_formulas, academic_collocations, academic_words, recurring_lemmas, level = tavr_text_analysis._get_data_for_web()
            trigrams_html = trigrams.to_html(index=False)
            stats_html = stats.to_html(index=False, header=False)
            academic_formulas_html = academic_formulas.to_html(index=False)
            academic_collocations_html = academic_collocations.to_html(index=False)
            academic_words_html = academic_words.to_html(index=False)
            recurring_lemmas_html = recurring_lemmas.to_html(index=False)

            # Original dictionary from TAVR
            self.marked_up_tokens = tavr_text_analysis.marked_up_tokens
            # Dictionary token.text as a key and id as a value
            for token in list(self.marked_up_tokens.keys()):
                self.token_id_dict[f'{self.marked_up_tokens[token]['id']}'] = token
            # List of tokens text to return it to javascript
            self.tokens_text = [token.text for token in list(self.marked_up_tokens.keys())]
            # Dictionary of tokens to return it to javascript
            tokens_for_json = dict()
            for key, value in tavr_text_analysis.marked_up_tokens.items():
                tokens_for_json[f'{key.text}'] = value
            with open('tmp/tokens.json', 'w') as f:
                json.dump(tokens_for_json, f, indent=2, sort_keys=False)

            return jsonify(table_trigrams=trigrams_html,
                           table_stats=stats_html,
                           table_academic_formulas=academic_formulas_html,
                           table_academic_collocations=academic_collocations_html,
                           table_academic_words=academic_words_html,
                           table_recurring_lemmas=recurring_lemmas_html,
                           level=level,
                           recurring_lemmas=list(recurring_lemmas['Lemma']),
                           len_academic_formulas=len(academic_formulas['Academic formula']),
                           len_academic_collocations=len(academic_collocations['Academic collocation']),
                           len_academic_words=len(academic_words['Academic word']))

        @self._app.route('/get_tokens', methods=['GET'])
        def get_tokens():
            with open('tmp/tokens.json', 'r') as f:
                tokens = orjson.loads(f.read())
            return jsonify(tokens, self.tokens_text)

        @self._app.route('/get_replacements', methods=['POST'])
        def get_replacements():
            response = request.get_json()
            id = response.get('data')
            token = self.token_id_dict[f'{id}']
            replacements, replacements_level = tavr.TokenReplacementOptions(self.marked_up_tokens).get_replacement_options(
                token, True
            )
            return jsonify(lemmas=replacements,
                           levels=replacements_level)

        if __name__ == '__main__':
            self._app.run(debug=True)


def main():
    Connection().connect()


if __name__ == '__main__':
    main()

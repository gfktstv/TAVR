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

            trigrams, stats, academic_formulas, recurring_lemmas, level = tavr_text_analysis._get_data_for_web()
            trigrams_html = trigrams.to_html(index=False)
            stats_html = stats.to_html(index=False, header=False)
            academic_formulas_html = academic_formulas.to_html(index=False)
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
            with open('temporary_files/tokens.json', 'w') as f:
                json.dump(tokens_for_json, f, indent=2, sort_keys=False)

            return jsonify(trigrams_html, stats_html, academic_formulas_html, recurring_lemmas_html, level)

        @self._app.route('/get_tokens', methods=['GET'])
        def get_tokens():
            with open('temporary_files/tokens.json', 'r') as f:
                tokens = json.load(f)
            return jsonify(tokens, self.tokens_text)

        @self._app.route('/get_replacements', methods=['POST'])
        def get_replacements():
            response = request.get_json()
            id = response.get('data')
            token = self.token_id_dict[f'{id}']
            replacements = tavr.TokenReplacementOptions(self.marked_up_tokens).get_replacement_options(
                token, True
            )
            return jsonify(replacements)

        if __name__ == '__main__':
            self._app.run(debug=True)


Connection().connect()

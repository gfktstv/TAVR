from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_cors import CORS
from tavr import *


app = Flask(__name__)
CORS(app)
app.app_context().push()


@app.route('/')
def index():
    return redirect(url_for('static', filename='web/tavr-main.html'))


@app.route('/receive_data', methods=['POST'])
def receive_data():
    """Receives a data from the web page"""
    # Get the JSON data from the request and extracts input data (essay)
    data = request.get_json()
    essay = data.get('data')
    measurements = LevelAndDescription(Text(essay)).description_dict

    return jsonify(measurements)


if __name__ == '__main__':
    app.run(debug=True)
.PHONY: all create-venv install create-temp-dir
all: install create-temp-dir

PYTHON := $(shell which python3 || which python)

create-venv:
	$(PYTHON) -m venv venv

install: create-venv
	venv/bin/pip install -r requirements
	venv/bin/python3 -m spacy download en_core_web_lg
	venv/bin/python3 -m nltk.downloader wordnet

create-temp-dir:
	mkdir -p ./tmp

clean:
	rm -rf ./tmp
	rm -rf ./venv
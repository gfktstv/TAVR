.PHONY: all install create-temp-dir
all: install create-temp-dir

PYTHON := $(shell which python3 || which python)

install:
	pip install -r requirements
	$(PYTHON) -m spacy download en_core_web_lg
	$(PYTHON) -m nltk.downloader wordnet

create-temp-dir:
	mkdir ./tmp
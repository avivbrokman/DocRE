#!/bin/bash

#navigate to root
# cd .

nohup python spacy_main.py fit --config ELRE_spacy_config.yaml > script_output.txt 2>&1

nohup python spacy_main.py fit --config ELRE_spacy_config1.yaml > script_output.txt 2>&1

nohup python spacy_main.py fit --config ELRE_spacy_config2.yaml > script_output.txt 2>&1

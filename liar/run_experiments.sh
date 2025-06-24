#!/bin/bash

# PT-BR EXPERIMENTS (GEMMA 3)
cd "pt-br/scripts"
python few_shot_all.py
python few_shot_only_statement.py
python zero_shot_all.py
python zero_shot_only_statement.py

cd "../.."


# ENGLISH EXPERIMENTS (GEMMA 3)
cd "english/scripts" 
python few_shot_all.py 
python few_shot_only_statement.py 
python zero_shot_all.py
python zero_shot_only_statement.py


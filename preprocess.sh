#! /bin/bash

mkdir -p models features predictions results

unzip data/data.zip

pip3 install -r requirements.txt
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.1/en_core_web_md-1.2.1.tar.gz

python3 -m spacy link en_core_web_md en_core_web_md

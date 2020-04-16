#!/usr/bin/env python -u
import sys, os
import argparse

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tag_en(sent):
    tokens = nltk.word_tokenize(sent)
    tagged = nltk.pos_tag(tokens)

    return [tag[1] for tag in tagged]

def main(args):
    tagger = {
        'en': pos_tag_en,
    }[args.lang]
    
    for sent in args.input:
        sent = sent.strip()
        tags = tagger(sent)
        print(sent, file=sys.stderr)
        print(*tags, sep=' ', file=args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  default=sys.stdin)
    parser.add_argument('-o', '--output', default=sys.stdout)
    parser.add_argument('-l', '--lang',   default='en', choices=['en', 'de', 'fr', 'cs'])

    args = parser.parse_args()
    assert args.lang == 'en'

    main(args)
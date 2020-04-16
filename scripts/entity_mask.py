#!/usr/bin/env python -u
import sys, os
import argparse
from tqdm import tqdm
import numpy as np
import json

# import flickr30k_entities script
TOOLS = os.path.realpath(os.path.dirname(__file__) + '/../tools')
sys.path.append(os.path.join(TOOLS, 'flickr30k_entities'))
from flickr30k_entities_utils import *

def info(s, **kwarg):
    print(s, file=sys.stderr, **kwarg)

def load_task1_data(task1_dir, split):
    tok_file = os.path.join(task1_dir, 'tok', '{}.lc.norm.tok.en'.format(split))
    pos_file = os.path.join(task1_dir, 'pos', '{}.lc.norm.tok.pos.en'.format(split))
    ord_file = os.path.join(task1_dir, 'image_splits', '{}.txt'.format(split))

    ord_data = [line.strip().split('.')[0] for line in open(ord_file)]

    data = {
        ord: {'sent': tok.strip().split(' '), 'pos': pos.strip().split(' ')}
        for tok, pos, ord in tqdm(zip(open(tok_file), open(pos_file), ord_data))
    }

    return data, ord_data

def load_entities_data(flickr30k_entities, ord_data):
    annotation_dir = os.path.join(flickr30k_entities, 'Annotations')

    def get_caps(ord):
        return get_sentence_data(os.path.join(flickr30k_entities, 'Sentences', '{}.txt'.format(ord)))
    def get_anno(ord):
        return get_annotations(os.path.join(flickr30k_entities, 'Annotations', '{}.xml'.format(ord)))

    data = {
        ord: {'caps': get_caps(ord), 'anno': get_anno(ord)}
        for ord in tqdm(ord_data)
    }

    return data

def token_match_ratio(s1, s2):
    s2 = s2.lower().split(' ')

    matches = sum([int(t1 == t2) for t1, t2 in zip(s1, s2)])
    ratio = (matches/len(s1) + matches/len(s2)) / 2
    return ratio

def mask_datum(datum, mask_tok='[v]'):
    masked = datum['sentence'].copy()
    poss = datum['pos']

    try:
        n_masked = 0
        for phrase in datum['phrases']:
            start_at = phrase['first_word_index']
            phrase_len = len(phrase['phrase'].split(' '))
            for i in range(start_at, start_at + phrase_len):
                if poss[i] == 'NN':
                    masked[i] = mask_tok
                    n_masked += 1
                    # mask only first NN
                    break
        error = False
    except:
        masked = datum['sentence'].copy()
        n_masked = 0
        error = True
    
    return ' '.join(masked), n_masked, error

def process_datum(datum, ent_datum):
    s1 = datum['sent']

    # find the caption used by multi30k
    fit_prob = [token_match_ratio(s1, c['sentence']) for c in ent_datum['caps']]
    bestfit_id = np.argmax(fit_prob)
    bestfit_prob = fit_prob[bestfit_id]
    bestfit = ent_datum['caps'][bestfit_id]

    info('< {} (matched: {:.2f})'.format(bestfit['sentence'], bestfit_prob*100))

    # remove unseen annotation
    seen_boxes = list(ent_datum['anno']['boxes'].keys())
    found = {
        'sentence': s1,
        'raw': bestfit['sentence'],
        'pos': datum['pos'],
        'phrases': [p for p in bestfit['phrases'] if p['phrase_id'] in seen_boxes],
        'match_ratio': bestfit_prob
    }

    # mask
    found['masked'], found['n_masked'], found['error'] = mask_datum(found)
    found['masked_ratio'] = found['n_masked'] / len(found['sentence'])

    print('{}'.format(found['masked']), file=args.output)
    
    return found

def main(args):
    data, ord_data = load_task1_data(args.task1, args.split)
    ent_data = load_entities_data(args.flickr30k_entities, ord_data)

    data = {
        ord: process_datum(data[ord], ent_data[ord])
        for ord in ord_data
    }

    # save data
    if args.data_json:
        info('Saving data ...')
        with open(args.data_json, 'w') as fp:
            json.dump({'data': data, 'ord': ord_data}, fp)

if __name__ == "__main__":
    task1 = os.path.realpath(os.path.dirname(__file__) + '/../data/task1')
    flickr30k_entities = os.path.realpath(os.path.dirname(__file__) + '/../tools/flickr30k_entities')

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default=sys.stdout)
    parser.add_argument('-d', '--data-json', default=None)

    parser.add_argument('-s', '--split', required=True, choices=['train', 'val', 'test_2016_flickr'])

    parser.add_argument('--task1', default=task1)
    parser.add_argument('--flickr30k-entities', default=flickr30k_entities)

    args = parser.parse_args()
    info(args)

    main(args)

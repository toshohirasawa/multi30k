#!/bin/bash -e

REPO=$(realpath $(dirname $0)/../)
TOK=${REPO}/data/task1/tok
POS=${REPO}/data/task1/pos
TAGGER=${REPO}/scripts/pos.py
LANG=en

mkdir -p ${POS}

for SPLIT in train val test_2016_flickr test_2017_flickr test_2018_flickr; do
    INFILE=${TOK}/${SPLIT}.lc.norm.tok.${LANG}
    OUTFILE=${POS}/${SPLIT}.lc.norm.tok.pos.${LANG}

    cat ${INFILE} | ${TAGGER} -l en | tee ${OUTFILE}
done

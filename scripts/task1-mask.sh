#!/bin/bash -e

REPO=$(realpath $(dirname $0)/../)
MASK=${REPO}/data/task1/mask
PARSER=${REPO}/scripts/entity_mask.py

mkdir -p ${MASK}

for SPLIT in test_2016_flickr val train; do
    OUTFILE=${MASK}/${SPLIT}.lc.norm.tok.entity.en
    DATAFILE=${MASK}/${SPLIT}.lc.norm.tok.entity.data
    ${PARSER} -s ${SPLIT} -d ${DATAFILE} | tee ${OUTFILE}
done

#!/usr/bin/env bash

source ~/.bashrc
conda activate uj 

set -xe 
export CUDA_VISIBLE_DEVICES=1

RERUN_ROOT=${1:-UNKNOWN}
MOV_POSTFIX=":${2:-run-0}"
export CUDA_VISIBLE_DEVICES=${3:-0}

RERUN_ROOT_MV="${RERUN_ROOT}${MOV_POSTFIX}"
mv $RERUN_ROOT $RERUN_ROOT_MV


for DIR in $RERUN_ROOT_MV/*/
do 
    echo RERUNNING $DIR
    source $DIR/rerun.sh
    echo ""
done
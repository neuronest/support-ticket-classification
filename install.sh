#!/usr/bin/env bash

rm -rf ./env

conda create -p ./env python=3.7.7
conda activate ./env

pip install -r requirements.txt
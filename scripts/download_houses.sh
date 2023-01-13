#!/bin/sh
mkdir data
cd data && \
    kaggle competitions download -c house-prices-advanced-regression-techniques  && \
    unzip house-prices-advanced-regression-techniques.zip && \
    rm house-prices-advanced-regression-techniques.zip
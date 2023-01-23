#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.


for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    gunzip -c ${fname}.gz > ${fname}
done

mv t10k-images-idx3-ubyte test-images-idx3-ubyte
mv t10k-labels-idx1-ubyte test-labels-idx1-ubyte


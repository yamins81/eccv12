#!/bin/bash
for DIR in thoreano Theano.git scikit-data eccv12 hyperopt ; do
    pushd ../$DIR
    git status

    popd
done

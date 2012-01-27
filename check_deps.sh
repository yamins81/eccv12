#!/bin/bash
for DIR in thoreano Theano scikit-data eccv12 hyperopt asgd ; do
    pushd ../$DIR
    echo '==========================================================='
    echo $DIR
    echo '==========================================================='

    git status -s

    popd
done

#!/bin/bash
for DIR in thoreano Theano scikit-data eccv12 hyperopt asgd ; do

    echo ''
    echo '==========================================================='
    echo $DIR
    echo '==========================================================='
    pushd ../$DIR > /dev/null

    git status -s

    popd > /dev/null
done

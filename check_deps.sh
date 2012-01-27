#!/bin/bash

DIRS="thoreano Theano scikit-data eccv12 hyperopt asgd genson"

if [ " st" = " $1" -o " " = " $1" ] ; then
    for DIR in $DIRS; do

        echo ''
        echo '==========================================================='
        echo $DIR
        echo '==========================================================='
        pushd ../$DIR > /dev/null

        git status -s

        popd > /dev/null
    done
fi

if [ " push" = " $1" ] ; then
    for DIR in $DIRS ; do

        echo ''
        echo '==========================================================='
        echo $DIR
        echo '==========================================================='
        pushd ../$DIR > /dev/null

        git push

        popd > /dev/null
    done
fi


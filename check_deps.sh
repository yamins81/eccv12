#!/bin/bash

DIRS="thoreano Theano scikit-data eccv12 hyperopt asgd genson"

if [ " st" = " $1" -o " " = " $1" ] ; then CMD="git status -s"
elif [ " push" = " $1" ] ; then            CMD="git push"
elif [ " pull" = " $1" ] ; then            CMD="git pull"
fi

if [ " $CMD" = " " ] ; then
    echo "Unrecognized argument: $1"
else
    for DIR in $DIRS; do

        echo ''
        echo '==========================================================='
        echo $DIR
        echo '==========================================================='
        pushd ../$DIR > /dev/null
        $CMD
        popd > /dev/null
    done
fi

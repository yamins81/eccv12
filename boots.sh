#!/bin/sh

set -x
set -e

pip install -r requirements.txt

git submodule init && git submodule update

SUBMODULES="asgd pyll pythor3 scikit-data thoreano hyperopt"
# simple_setup not for installation

for SM in SUBMODULES ; do
    (cd $SM && python setup.py develop)
done

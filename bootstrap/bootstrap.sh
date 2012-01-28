#!/bin/bash

ROOT=${HOME}/venv/eccv12

git_up() {
    rep=$1
    branch=$2
    dir=$3
    cd ${ROOT}
    (git clone -b $branch $rep $dir)
    cd $dir
    git pull
    pip install -e .
}

set -e
set -x

if [[ ! -d ${ROOT} ]]; then
    mkdir -p ${ROOT};
    virtualenv --system-site-packages ${ROOT};
fi;

source ${HOME}/venv/eccv12/bin/activate

# -- dependencies
pip install -vUI ipython nose lockfile

pip install -v pymongo

pip install -v theano

git_up git@github.com:nsf-ri-ubicv/pythor3.git develop pythor3

git_up git@github.com:nsf-ri-ubicv/thoreano.git master thoreano

git_up https://github.com/jaberg/scikit-data.git master scikit-data

git_up https://github.com/npinto/hyperopt.git master hyperopt

git_up https://github.com/npinto/MonteTheano.git master montetheano

git_up https://github.com/npinto/asgd.git jaberg/master asgd

git_up https://github.com/davidcox/genson.git yamins81/feature/hyperopt genson

# -- eccv12
git_up git@github.com:nsf-ri-ubicv/boosted_hyperopt_eccv12.git fson eccv12

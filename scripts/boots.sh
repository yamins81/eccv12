VENV=${HOME}/.VENV/eccv12

set -e
set -x

test ! -d ${VENV}
mkdir -p ${VENV}
virtualenv --no-site-packages ${VENV}
source ${VENV}/bin/activate

SRC=${VENV}/src
mkdir ${SRC}
cd ${SRC}

git clone git@github.com:nsf-ri-ubicv/boosted_hyperopt_eccv12.git eccv12
# -- these are submodules of boosted_hyperopt_eccv12
# they must be installed in the following order:
# $(cd pyll && python setup.py develop )
# $(cd hyperopt && python setup.py develop )

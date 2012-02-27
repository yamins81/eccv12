#!/bin/bash

### ADD THIS TO fabfile.py, construct dbname automatically

export SCIKIT_DATA=/scratch_local/skdata
L=$SCIKIT_DATA/lfw/aligned
mkdir -p $L
rsync -a ~/.skdata/lfw/aligned/ $L/

. VENV/eccv12/bin/activate
VENV/eccv12/src/eccv12/hyperopt/bin/hyperopt-mongo-worker \
    --mongo=honeybadger:44556/try_no_inj \
    --workdir=/scratch_local/eccv12.workdir

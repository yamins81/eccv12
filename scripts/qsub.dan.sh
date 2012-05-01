#!/bin/bash

### ADD THIS TO fabfile.py, construct dbname automatically

export SCIKIT_DATA=/scratch_local/$USER/.skdata
rsync -a ~/.skdata/PubFig83 $SCIKIT_DATA/

DBNAME=mar2_pubfig83

VENV=~/eccv12
. $VENV/bin/activate
$VENV/boosted_hyperopt_eccv12/hyperopt/bin/hyperopt-mongo-worker \
    --mongo=honeybadger:44556/$DBNAME \
    --workdir=/scratch_local/$USER/eccv12.workdir

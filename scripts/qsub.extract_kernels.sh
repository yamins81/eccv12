#!/bin/bash

### ADD THIS TO fabfile.py, construct dbname automatically

export SCIKIT_DATA=/scratch_local/skdata
L=$SCIKIT_DATA/lfw/aligned
mkdir -p $L
rsync -a ~/.skdata/lfw/aligned/ $L/

. VENV/eccv12/bin/activate
cd VENV/eccv12/src/eccv12
#'/home/dyamins/eccv12/boosted_hyperopt_eccv12/Temp/simple_mix_id_nf.pkl'
#IDS_FILENAME=/home/dyamins/eccv12/boosted_hyperopt_eccv12/Temp/ada_mix_id_nf.pkl
#fab extract_kernel:$IDS_FILENAME
fab extract_kernel_par_tpe:$PBS_ARRAYID

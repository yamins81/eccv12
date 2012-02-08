#!/bin/bash

# -- DEBUG
#echo ${PBS_GPUFILE} >> /dev/stderr
#cat ${PBS_GPUFILE} >> /dev/stderr
#wc -l ${PBS_GPUFILE} >> /dev/stderr

if [[ " " -eq "${PBS_GPUFILE} " ]] ; then
    echo Not running under PBS >> /dev/stderr
    exit 0
else
    N_GPUS=$(grep -e '.*-gpu.*' ${PBS_GPUFILE} | wc -l)
    # echo 'n gpus' $N_GPUS >> /dev/stderr
fi

# GO !
if [[ ! ${N_GPUS} -eq 1 ]]; then
    echo "WRONG NUMBER OF GPUS!!!!" >> /dev/stderr
    echo " "  # return something for caller not to hang
    exit 1;
fi;

gpu=$(sed -e 's/.*-gpu\(.*\)/\1/g' ${PBS_GPUFILE})
echo "My gpu is ${gpu}" >> /dev/stderr
echo ${gpu}

#!/bin/bash
# Script to be used in the Cobalt_correlator_regression run
# used in combination with the parset 3c169.parset
# files are written to the temp dir of the jenkins account

# enable dumping core
ulimit -c unlimited

MODULELIST="prun sge gcc openmpi/gcc cuda50/toolkit cuda50/fft"
module load $MODULELIST

source /home/jenkins/jenkins/Cobalt_correlator_regression/gnu_opt/installed/lofarinit.sh

mpirun -np 1 -H localhost /home/jenkins/jenkins/Cobalt_correlator_regression/gnu_opt/installed/bin/rtcp /var/scratch/jenkins/parsets/3c169.parset  2>&1 | tee /var/scratch/jenkins/output_data_correlator/last_run.log
exit ${PIPESTATUS[0]}
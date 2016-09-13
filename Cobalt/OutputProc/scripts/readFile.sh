#!/bin/sh

# This script takes the output file of the TFlopCorrelator
# splits it and copies the file containing the raw data to 
# the current directory
# it is only meant to be used for the CDR

MSNAME=ObservationA.MS
NEWFILE=vis.dat.`date +%F_%k:%M`

export PATH=`pwd`:$PATH
. /lofarbuild/aips++/dev/aipsinit.sh
echo "mssplit('$MSNAME', 1); exit" | glish -l ../libexec/glish/mssplit.g
cp ${MSNAME}_p1/vis.dat $NEWFILE
echo
echo "created $NEWFILE"


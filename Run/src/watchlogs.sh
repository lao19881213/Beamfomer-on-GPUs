#!/bin/bash

source locations.sh

FLAGS="-n 10000"

CNPROC_LOG=$LOGDIR/CNProc.log
IONPROC_LOG=$LOGDIR/IONProc.log

echo Reading logs from $LOGDIR
echo Reading multitail configuration from $ETCDIR

ERRORLOGS=

if [ $ISPRODUCTION -eq 1 ]
then
  for l in $LOGDIR/BlueGeneControl.log $LOGDIR/startBGL.log
  do
    echo Reading additional error log $l
    ERRORLOGS="$ERRORLOGS $FLAGS -cS olap -fr errors -I $l"
  done 
fi

multitail --no-mark-change --follow-all --retry-all -m 10240 --basename -F $ETCDIR/multitail-olap.conf \
  $FLAGS -t "-- FLAGS --"  -fr flags -ks flags -i $IONPROC_LOG \
  $FLAGS -t "-- ERRORS --" -fr errors          -i $IONPROC_LOG \
  $FLAGS                   -fr errors          -I $CNPROC_LOG \
  $ERRORLOGS \
  $FLAGS -t "IONProc/Storage"                  -i $IONPROC_LOG \
  $FLAGS -t "CNProc"       -wh 5               -i $CNPROC_LOG


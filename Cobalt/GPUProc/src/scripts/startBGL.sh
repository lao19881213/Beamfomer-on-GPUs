#!/bin/bash
# startBGL.sh jobName executable workingDir parset observationID
#
# jobName
# executable      executable file (should be in a place that is readable from BG/L)
# workingDir      directory for output files (should be readable by BG/L)
# parset          the parameter file
# observationID   observation number
#
# This script is called by OnlineControl to start an observation.

if test "$LOFARROOT" == ""; then
  echo "LOFARROOT is not set! Exiting."
  exit 1
fi

PARSET="$4"
OBSID="$5"

# The file to store the PID in
PIDFILE=$LOFARROOT/var/run/rtcp-$OBSID.pid

# The file we will log the observation output to
LOGFILE=$LOFARROOT/var/log/rtcp-$OBSID.log

(
# Always print a header, to match errors to observations
echo "---------------"
echo "now:      " `date +"%F %T"`
echo "called as: $0 $@"
echo "pwd:       $PWD"
echo "LOFARROOT: $LOFARROOT"
echo "obs id:    $OBSID"
echo "parset:    $PARSET"
echo "log file:  $LOGFILE"
echo "---------------"

function error {
  echo "$@"
  exit 1
}

[ -n "$PARSET" ] || error "No parset provided"
[ -f "$PARSET" -a -r "$PARSET" ] || error "Cannot read parset: $PARSET"

# Start observation in the background
runObservation.sh "$PARSET" > $LOGFILE 2>&1 </dev/null &
PID=$!
echo "PID: $PID"

# Keep track of PID for stop script
echo "PID file: $PIDFILE"
echo $PID > $PIDFILE || error "Could not write PID file: $PIDFILE"

# Done
echo "Done"

) 2>&1 | tee -a $LOFARROOT/var/log/startBGL.log

# Return the status of our subshell, not of tee
exit ${PIPESTATUS[0]}


#!/bin/bash

# A mock OnlineControl to receive the observation status
# and the LTA feedback.

# 1. Read parset
PARSET="$1"

function error {
  echo "$@"
  exit 1
}

function getkey {
  KEY=$1
  <$PARSET perl -ne '/^'$KEY'\s*=\s*"?(.*?)"?\s*$/ || next; print "$1";'
}

[ -n "$PARSET" ] || error "No parset specified"
[ -r "$PARSET" ] || error "Cannot read parset: $PARSET"

# 2. Open port @ 21000 + 1000 % obsid
OBSID=`getkey Observation.ObsID`
RESULT_PORT=$((21000 + $OBSID % 1000))

STATUSSTR=`timeout 30s nc -l $RESULT_PORT`

# 3. Read string: ABORT or FINISHED
if [ "$STATUSSTR" == "FINISHED" ]
then
  echo "Observation reported success"

  # 4. If finished, check for existence of feedback file
  FEEDBACK_FILE=`getkey Cobalt.Feedback.remotePath`/Observation${OBSID}_feedback

  # Check existence and access rights
  [ -e $FEEDBACK_FILE ] || error "Feedback file not found: $FEEDBACK_FILE"
  [ -r $FEEDBACK_FILE ] || error "Feedback file not readable: $FEEDBACK_FILE"
  [ -f $FEEDBACK_FILE ] || error "Feedback file not a regular file: $FEEDBACK_FILE"

  # Check file size
  FILESIZE=`stat -c %s $FEEDBACK_FILE`
  [ $FILESIZE -ne 0 ] || error "Feedback file empty: $FEEDBACK_FILE"
elif [ "$STATUSSTR" == "ABORT" ]
then
  echo "Observation reported failure"
else
  error "Invalid status string: '$STATUSSTR'"
fi

exit 0


#!/bin/bash

source locations.sh

function start() {
  set_psetinfo

  # make sure the log dir exists
  mkdir -p "$LOGDIR"

  TMPDIR="`mktemp -d`"
  PIDFILE="$TMPDIR/pid"

  # use a fifo to avoid race conditions
  mkfifo "$PIDFILE"

  (mpirun -noallocate -mode VN -partition "$PARTITION" -env DCMF_COLLECTIVES=0 -env BG_MAPPING=XYZT -env LD_LIBRARY_PATH=/bgsys/drivers/ppcfloor/comm/lib:/bgsys/drivers/ppcfloor/runtime/SPI:/globalhome/romein/lib.bgp -cwd "$RUNDIR" -exe "$CNPROC" 2>&1 &
  echo $! > "$PIDFILE") | LOFAR/Logger.py $LOGPARAMS "$LOGDIR/CNProc.log" &

  PID=`cat "$PIDFILE"`
  rm -f "$PIDFILE"
  rmdir "$TMPDIR"

  if [ -z "$PID" ]
  then
    PID=DOWN
  fi   
}

function stop() {
  set_psetinfo

  # graceful exit
  alarm 10 gracefullyStopBGProcessing.sh

  # ungraceful exit
  [ -e /proc/$PID ] && (
    # mpikill only works when mpirun has started running the application
    mpikill "$PID" ||

    # ask DNA to kill the job
    (cd /;bgjobs -u $USER -s | awk "/$PARTITION/ { print \$1; }" | xargs -L 1 bgkilljob) ||

    # kill -9 is the last resort
    kill -9 "$PID"
  ) && sleep 10

  # wait for job to die
  TIMEOUT=10

  while true
  do
    JOBSTATUS=`cd /;bgjobs -u $USER -s | awk "/$PARTITION/ { print \\$6; }"`
    JOBID=`cd /;bgjobs -u $USER -s | awk "/$PARTITION/ { print \\$1; }"`

    if [ -z "$JOBID" ]
    then
      # job is gone
      break
    fi  

    case "$JOBSTATUS" in
      dying)
        sleep 1
        continue ;;

      running)
        sleep 1

        if [ $((--TIMEOUT)) -ge 0 ]
        then
          continue
        fi
        ;;
    esac

    echo "Failed to kill BG/P job $JOBID. Status is $JOBSTATUS"
    break
  done
}

. controller.sh

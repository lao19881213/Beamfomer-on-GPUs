#!/bin/bash
# stopAP.sh partition jobName
#
# jobName       The name of the job
# observationID Observation number
#

#echo -n "Killing job " $2
#killjob $1 `cat $2.jobID`
#rm -f $2.pid $2.ps

source /opt/lofar/bin/locations.sh

JOB=$1
OBSID=$2

# stopBGL.sh is used both to abort an observation and at the end of
# an observation. Sleep a bit so that OLAP can temrinate on its own
# in the latter case.

if [ $ROUTE_TO_COBALT -eq 1 ]
then
  # Reroute to Cobalt
  echo "Rerouting to Cobalt"

  # Stop the observation on Cobalt
  # (TODO: disabled because MAC calls stopBGL before we're finished)
  #ssh mol@10.168.96.1 stopBGL.sh 1 $OBSID

  # And.. done!
  exit 0
fi

/opt/lofar/bin/commandOLAP.py -P $PARTITION cancel $OBSID

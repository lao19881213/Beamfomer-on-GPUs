#!/bin/bash
CONFIG=/opt/lofar/etc/BlueGeneControl.conf

. $CONFIG

SCRIPTDIR=$BINPATH/LOFAR
RASLOG=/bgsys/logs/BGP/bgsn-bgdb0-mmcs_db_server-current.log

# ##### HARDWARE #####

function reachable {
  IP=$1

  ping $IP -c 1 -w 2 -q >/dev/null 2>&1
  if [ $? -eq 0 ]
  then
    echo yes
  else  
    echo no
  fi
}

# ----- Partition information

echo partition $PARTITION

PARTITION_STATUS=`bgpartstatus $PARTITION`
PARTITION_OWNER=`bgbusy | grep $PARTITION | awk '{ print $5; }'`

echo '# partition status should be "busy"'
echo partition_status $PARTITION_STATUS

echo '# partition owner should be "'$USER'"'
echo partition_owner $PARTITION_OWNER

# ----- I/O node information

IONODES=`$SCRIPTDIR/Partitions.py -l $PARTITION`

echo '# a list of I/O node IP addresses'
echo ionode_list $IONODES

echo '# all I/O nodes should be reachable'
for i in $IONODES
do
  echo ionode_reachable $i `reachable $i`
  echo ionode_mac $i `/sbin/arp -n $i | awk '/ether/ { print $3; }'`
done

echo '# the service node (bgsn) should be reachable'
echo service_node_reachable `reachable bgsn`

# ----- RAS events

# can't access logs on bgfen, cheat using an I/O node
echo '# the latest RAS events for this partition and the current owner'
FIRST_IONODE=`echo $IONODES | awk '{ print $1; }'`
ssh -q $FIRST_IONODE grep RasEvent $RASLOG 2>&1 | grep $PARTITION_OWNER:$PARTITION | perl -ne '
/time="([^"]+)".*BG_LOC="([^"]+)".*BG_MSG="([^"]+)".*BG_SEV="([^"]+)"/ || next;
print "rasevent $2 $1 $4 $3\n";
'

# ##### SOFTWARE #####

# ----- Job information

JOB_STATUS_LONG=`bgjobs -s -u $PARTITION_OWNER | grep $PARTITION`

if [ "$JOB_STATUS_LONG" == "" ]
then
  # no job running
  JOB_NAME=none
  JOB_STATUS=none
else
  JOB_NAME=`echo $JOB_STATUS_LONG | awk '{ print $4; }'`
  JOB_STATUS=`echo $JOB_STATUS_LONG | awk '{ print $6; }'`
fi

echo '# job name should be "CN_Processing"'
echo cn_job_name $JOB_NAME

echo '# job status should be "running"'
echo cn_job_status $JOB_STATUS


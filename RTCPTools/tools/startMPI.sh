# startMPI.sh jobName machinefile executable paramfile noNodes
#
# $1 jobName             identifier for this job
# $2 machinefile         procID.machinefile
# $3 executable          processname
# $4 parameterfile       procID.ps
# $5 numberOfNodes
#
# calls mpirun and remembers the pid
#

# now all ACC processes expect to be started with ACC as first parameter

# start process
# TODO: on some hosts, mpirun has a different name (or a specific path)
#       on some hosts, we should use -hostfile instead of -machinefile

partition=`Run/src/getPartition.py --parset=Storage.parset`
stationlist=`Run/src/getStations.py --parset=Storage.parset`
clock=`Run/src/getSampleClock.py --parset=Storage.parset`
integrationtime=`Run/src/getIntegrationtime.py --parset=Storage.parset`

Run/src/Run.Storage.py --partition=$partition --stationlist=$stationlist --integrationtime=$integrationtime --clock=$clock 1 >/dev/null 2>&1 &

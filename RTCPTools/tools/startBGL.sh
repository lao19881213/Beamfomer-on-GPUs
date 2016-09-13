# startBGL.sh jobName partition executable workingDir paramfile noNodes
#
# jobName
# partition
# executable      executable file (should be in a place that is readable from BG/L)
# workingDir      directory for output files (should be readable by BG/L)
# parameterfile   jobName.ps
# noNodes         number of nodes of the partition to use
#
# start the job and stores the jobID in jobName.jobID
#
# all ACC processes expect to be started with "ACC" as first parameter

# start process

partition=`Run/src/getPartition.py --parset=CNProc.parset`
stationlist=`Run/src/getStations.py --parset=CNProc.parset`
clock=`Run/src/getSampleClock.py --parset=CNProc.parset`
integrationtime=`Run/src/getIntegrationtime.py --parset=CNProc.parset`

Run/src/Run.CNProc.py --partition=$partition --stationlist=$stationlist --integrationtime=$integrationtime --clock=$clock 1 >/dev/null 2>&1 &

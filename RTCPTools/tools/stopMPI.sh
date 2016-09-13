# stopMPI.sh execName 
#
#
# Stops the given process by killing the process whose pid is in the
# proces.pid file.

# TODO: for some mpi versions it is not enough to kill mpirun
#       we could "killall executable", but that would also kill
#       processes started by another ApplicationController

ssh listfen cexec :0-1 killall -9 Storage orted

ssh listfen killall -9 mpirun

rm -f $1*.ps

pid=`ps -ef |grep '\-[w]dir' |grep -v 'sh \-c'|awk '{ print $2 }'`
if [ "${pid}" != "" ]; then
  kill -9 ${pid}
else
  echo 'no process to killed'  
fi

# stopAP.sh partition jobName
#
# partition BG/L partition the job is running on
# jobName   The name of the job
#

killall -9 mpirun

pid=`ps -ef |grep '\-[w]dir' |grep -v 'sh \-c'|awk '{ print $2 }'`
if [ "${pid}" != "" ]; then
  kill -9 ${pid}
else
  echo 'no process to killed'  
fi

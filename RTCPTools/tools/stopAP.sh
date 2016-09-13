# stopAP.sh partition jobName
#
# partition BG/L partition the job is running on
# jobName   The name of the job
#

for i in 33 34 37 38 41 42 45 46 49 50 53 54 57 58 61 62; do ssh 10.170.0.$i killall -9 IONProc; done; 

pid=`ps -ef |grep '\-[w]dir' |grep -v 'sh \-c'|awk '{ print $2 }'`
if [ "${pid}" != "" ]; then
  kill -9 ${pid}
else
  echo 'no process to killed'  
fi

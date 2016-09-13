#!/bin/bash
#
# CorrAppl: a start/stop/status script for swlevel
#
# Copyright (C) 2007
# ASTRON (Netherlands Foundation for Research in Astronomy)
# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Syntax: CorrAppl start|stop|status
#
# $Id: RTCP.sh 20433 2012-03-14 08:11:03Z overeem $
#

#
# SyntaxError msg
#
SyntaxError()
{
	Msg=$1

	[ -z "${Msg}" ] || echo "ERROR: ${Msg}"
	echo ""
	echo "Syntax: $(basename $0) start | stop | status"
	echo ""
	exit 1
}

#
# Start the program when it exists
#
start_prog()
{
	# put here your code to start your program
	echo 'start_prog()'
}

#
# Stop the program when it is running
#
stop_prog()
{
	# put here your code to stop your program
	killall ApplController
	ps -ef | grep -v grep | grep -v ACDaemon[^\ ] | grep ACDaemon 2>&1 >/dev/null
	if [ $? -ne 0 ]; then
	  if [ -f ../etc/ACD.admin ]; then 	
	    rm ../etc/ACD.admin
	  fi
	fi  
	echo ''
	echo 'Terminate the job and to free the resources'
	echo 'occupied by this job.'
	killall -9 mpirun
	echo ''
	echo 'Terminate IONProc on the IONodes.'
	for i in 33 34 37 38 41 42 45 46 49 50 53 54 57 58 61 62; do ssh 10.170.0.$i killall -9 IONProc; done;
	echo ''
	echo 'Terminate Storage on the StorageNodes.'
	ssh listfen cexec :0,1 killall -9 Storage orted
	echo ''
	echo 'Terminate mpirun (listfen).'
	ssh listfen killall -9 mpirun
	echo ''
	echo 'Killed old process.'
	pid=`ps -ef |grep '\-[w]dir' |grep -v 'sh \-c'|awk '{ print $2 }'`
	if [ "${pid}" != "" ]; then
          kill -9 ${pid}
	else
	  echo 'no process to killed'  
        fi	
}

#
# show status of program
#
# arg1 = levelnr
#
status_prog()
{
	# put here code to figure out the status of your program and
	# fill the variables prog and pid with the right information

	levelnr=$1
        # status: Storage
	prog='Storage'
	
	firstTime=1
	pid=DOWN
	for i in 1 2; do
	  ssh list00$i ps -C Storage 2>$1 1>/dev/null
	  if [ $? -eq 0 ]; then
	    if [ ${firstTime} -eq 1 ]; then
	      firstTime=0
	      prog=${prog}'(list00'$i
	      pid=`ssh list00$i ps --no-headers -C Storage | awk '{ print $1 }'`
	    else
	      prog=${prog}',list00'$i
	      pid=${pid}':'`ssh list00$i ps --no-headers -C Storage | awk '{ print $1 }'`
	    fi
	  fi
	done

	if [ ${firstTime} -eq 0 ]; then
	  prog=${prog}')'
	fi
	
	echo ${levelnr} ${prog} ${pid} | awk '{ printf "%s : %-25s %s\n", $1, $2, $3 }'
	# status: IONProc
	prog='IONProc'
	firstTime=1
	pid=DOWN
	cnpid=DOWN
	
	for i in 33 34 37 38 41 42 45 46 49 50 53 54 57 58 61 62; do
	  ssh '10.170.0.'$i ps --no-headers -C IONProc 2>$1 1>/dev/null
	  if [ $? -eq 0 ]; then
	    if [ ${firstTime} -eq 1 ]; then
	      firstTime=0
	      prog=${prog}'(10.170.0.'$i
	      pid=`ssh '10.170.0.'$i ps --no-headers -C IONProc | awk '{ print $1 }'`
	    else
	      prog=${prog}',10.170.0.'$i
	      pid=${pid}':'`ssh '10.170.0.'$i ps --no-headers -C IONProc | awk '{ print $1 }'`
	    fi
	  fi
	done
	
	if [ ${firstTime} -eq 0 ]; then
	  prog=${prog}')'
	  cnpid=UP
	fi
	echo ${levelnr} ${prog} ${pid} | awk '{ printf "%s : %-25s %s\n", $1, $2, $3 }'     
	
	# status: CNProc
	prog='CNProc'
	echo ${levelnr} ${prog} ${cnpid} | awk '{ printf "%s : %-25s %s\n", $1, $2, $3 }'
	# this line should be left in, it shows the status in the right format
	#echo ${levelnr} ${prog} ${pid} | awk '{ printf "%s : %-25s %s\n", $1, $2, $3 }'
	#echo ${levelnr} ${prog} ${status} | awk '{ printf "%s : %-25s %s\n", $1, $2, $3 }'
	#echo ${levelnr} ${prog} `ssh $USER@bglsn /opt/lofar/bin/stopBGL.py --status=true` | awk '{ printf "%s : %-25s %s\n", $1, $2, $3 }'
}

#
# MAIN
#

# when no argument is given show syntax error.
if [ -z "$1" ]; then
	SyntaxError
fi

# first power down to this level
case $1 in
	start)	start_prog
			;;
	stop)	stop_prog
			;;
	status)	status_prog $2
			;;
	*)		SyntaxError
			;;
esac

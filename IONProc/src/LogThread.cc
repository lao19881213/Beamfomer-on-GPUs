//#  LogThread.cc:
//#
//#  Copyright (C) 2008
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: LogThread.cc 22267 2012-10-09 13:28:03Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <LogThread.h>
#include <Scheduling.h>
#include <Interface/PrintVector.h>
#include <Common/LofarLogger.h>
#include <Common/Thread/Cancellation.h>

#include <algorithm>
#include <cstdio>
#include <sstream>

#include <unistd.h>


namespace LOFAR {
namespace RTCP {


// log from separate thread, since printing from a signal handler causes deadlocks

LogThread::LogThread(unsigned nrRspBoards, std::string stationName)
:
  itsCounters(nrRspBoards),
  itsStationName(stationName)
{
}


void LogThread::start()
{
  itsThread = new Thread(this, &LogThread::mainLoop, "[LogThread] ", 65536);
}


LogThread::~LogThread()
{
  if (itsThread)
    itsThread->cancel();

  LOG_DEBUG_STR("[LogThread] finished");
}


#if defined HAVE_BGP_ION

bool LogThread::readCPUstats(struct CPUload &load)
{
  FILE *file = fopen("/proc/stat", "r");
  int  retval;

  if (file == 0)
    return false;

  // make sure the file is always closed -- even on cancellation (fscanf CAN be a cancellation point)
  struct D {
    ~D() {
      fclose(file);
    }

    FILE *file;
  } onDestruct = { file };
  (void)onDestruct;

  do
    retval = fscanf(file, "cpu %llu %*u %llu %llu %*u %*u %llu %*u\n", &load.user, &load.system, &load.idle, &load.interrupt);
  while (retval != 4 && retval != EOF);

  do
    retval = fscanf(file, "cpu0 %*u %*u %*u %llu %*u %*u %*u %*u\n", &load.idle0);
  while (retval != 1 && retval != EOF);

  return retval != EOF;
}


void LogThread::writeCPUstats(std::stringstream &str)
{
  struct CPUload load;
  struct timeval tv;

  static size_t lowIdleCount = 0;

  if (readCPUstats(load)) {
    gettimeofday( &tv, 0 );

    float timediff = (tv.tv_sec - previousTimeval.tv_sec) + (tv.tv_usec - previousTimeval.tv_usec)/1.0e6;

    unsigned idle0 = static_cast<unsigned>((load.idle0 - previousLoad.idle0) / timediff);
    if (idle0 < 10)
      lowIdleCount++;
    else
      lowIdleCount = 0;

    // TODO: Don't print this error in non-realtime mode
    if (lowIdleCount == 5)
      LOG_ERROR("CPU load critical on core 0");

    //str << ", us/sy/in/id: ["
    str << ", us/sy/in/id(0): ["
	<< fixed << setprecision(0) 
        << (unsigned(load.user	    - previousLoad.user)      + 2) / 4 / timediff << '/'
	<< (unsigned(load.system    - previousLoad.system)    + 2) / 4 / timediff << '/'
	<< (unsigned(load.interrupt - previousLoad.interrupt) + 2) / 4 / timediff << '/'
	<< (unsigned(load.idle	    - previousLoad.idle)      + 2) / 4 / timediff << '('
	<< idle0 << ")]";
#if 0
	<< "], id: ["
	<< (unsigned(load.idlePerCore[0] - previousLoad.idlePerCore[0]) << '/'

    for (unsigned cpu = 0; cpu < 4; cpu ++)
      str << unsigned(load.idle[cpu] - previousLoad.idle[cpu])
	  << (cpu == 3 ? ']' : ',');
#endif

    previousLoad = load;
    previousTimeval = tv;
  } else {
    str << ", no CPU load info";
  }
}

#endif


void LogThread::mainLoop()
{
#if defined HAVE_BGP_ION
  //doNotRunOnCore0();
  runOnCore0();
  readCPUstats(previousLoad);
  gettimeofday(&previousTimeval,0);
#endif

  //LOG_DEBUG("LogThread running");

  // non-atomic updates from other threads cause race conditions, but who cares

  while (true) {
    std::stringstream	  logStr;
    std::vector<unsigned> counts(itsCounters.size());

    for (unsigned rsp = 0; rsp < itsCounters.size(); rsp ++) {
      counts[rsp]		= itsCounters[rsp].received;
      itsCounters[rsp].received = 0;
    }

    logStr << "[station " << itsStationName << "] ";

    logStr << "received packets = " << counts;

    for (unsigned rsp = 0; rsp < itsCounters.size(); rsp ++) {
      counts[rsp]	       = itsCounters[rsp].badSize;
      itsCounters[rsp].badSize = 0;
    }

    if (static_cast<unsigned>(std::count(counts.begin(), counts.end(), 0U)) != counts.size())
      logStr << ", bad size = " << counts;

    for (unsigned rsp = 0; rsp < itsCounters.size(); rsp ++) {
      counts[rsp]		    = itsCounters[rsp].badTimeStamp;
      itsCounters[rsp].badTimeStamp = 0;
    }

    if (static_cast<unsigned>(std::count(counts.begin(), counts.end(), 0U)) != counts.size())
      logStr << ", bad timestamps = " << counts;

#if defined HAVE_BGP_ION
    writeCPUstats(logStr);
#endif

    LOG_INFO_STR(logStr.str());
    sleep(15);
  }

  //LOG_DEBUG("LogThread stopped");
}

} // namespace RTCP
} // namespace LOFAR

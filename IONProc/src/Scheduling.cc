//#  Scheduling.cc:
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
//#  $Id: Scheduling.cc 13340 2009-05-28 08:47:40Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#if defined HAVE_BGP_ION

#include <IONProc/Scheduling.h>
#include <Common/LofarLogger.h>

#include <iostream>
#include <cstdio>

#include <sched.h>


namespace LOFAR {
namespace RTCP {

void doNotRunOnCore0()
{
  cpu_set_t cpu_set;

  CPU_ZERO(&cpu_set);

  for (unsigned cpu = 1; cpu < 4; cpu ++)
    CPU_SET(cpu, &cpu_set);

  if (sched_setaffinity(0, sizeof cpu_set, &cpu_set) != 0) {
    LOG_WARN("sched_setaffinity failed");
    perror("sched_setaffinity");
  }
}


void runOnCore0()
{
  cpu_set_t cpu_set;

  CPU_ZERO(&cpu_set);
  CPU_SET(0, &cpu_set);

  if (sched_setaffinity(0, sizeof cpu_set, &cpu_set) != 0) {
    LOG_WARN("sched_setaffinity failed");
    perror("sched_setaffinity");
  }
}


void setPriority(unsigned priority)
{
  // priority 0: non-real time
  // priority 1-99: real time
  struct sched_param sched_param;

  sched_param.sched_priority = priority;

  if (pthread_setschedparam(pthread_self(), priority ? SCHED_RR : SCHED_OTHER, &sched_param) < 0)
    perror("pthread_setschedparam");
}

} // namespace RTCP
} // namespace LOFAR

#endif

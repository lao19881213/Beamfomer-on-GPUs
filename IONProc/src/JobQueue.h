//#  JobQueue.h
//#
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
//#  $Id: ION_main.cc 15296 2010-03-24 10:19:41Z romein $


//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#if !defined LOFAR_RTCP_JOB_QUEUE_H
#define LOFAR_RTCP_JOB_QUEUE_H

#include <Common/Thread/Condition.h>
#include <Common/Thread/Mutex.h>

#include <vector>


namespace LOFAR {
namespace RTCP {


class Job;

class JobQueue
{
  public:
    void		insert(Job *), remove(Job *);

    void		cancel(unsigned observationID);
    void		cancelAll();

    void		listJobs() const;

    void		waitUntilAllJobsAreFinished();

  private:
    friend class Job;

    std::vector<Job *>	itsJobs;

    mutable Mutex	itsMutex;
    Condition		itsReevaluate;
};


extern JobQueue jobQueue;


} // namespace RTCP
} // namespace LOFAR

#endif

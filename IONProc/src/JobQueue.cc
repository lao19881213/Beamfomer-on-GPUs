//#  JobQueue.cc:
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

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <Job.h>

#include <algorithm>



namespace LOFAR {
namespace RTCP {


JobQueue jobQueue;


void JobQueue::insert(Job *job)
{
  ScopedLock scopedLock(itsMutex);
  itsJobs.push_back(job);
}


void JobQueue::remove(Job *job)
{
  ScopedLock scopedLock(itsMutex);
  itsJobs.erase(find(itsJobs.begin(), itsJobs.end(), job));
  itsReevaluate.broadcast();
}


void JobQueue::cancel(unsigned observationID)
{
  ScopedLock scopedLock(itsMutex);

  for (std::vector<Job *>::iterator job = itsJobs.begin(); job != itsJobs.end(); job ++)
    if ((*job)->itsObservationID == observationID) {
      (*job)->cancel();
      return;
    }

  LOG_WARN_STR("[obs " << observationID << "] Could not cancel: not found");
}


void JobQueue::cancelAll()
{
  ScopedLock scopedLock(itsMutex);

  for (std::vector<Job *>::iterator job = itsJobs.begin(); job != itsJobs.end(); job ++)
    (*job)->cancel();
}


void JobQueue::listJobs() const
{
  ScopedLock scopedLock(itsMutex);

  for (std::vector<Job *>::const_iterator job = itsJobs.begin(); job != itsJobs.end(); job ++)
    (*job)->printInfo();
}


} // namespace RTCP
} // namespace LOFAR

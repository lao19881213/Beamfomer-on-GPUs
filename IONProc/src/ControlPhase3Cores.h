//#  ControlPhase3Cores.h: Send PROCESS commands to dedicated phase 3 cores
//#
//#  Copyright (C) 2006
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
//#  $Id: ControlPhase3Cores.h 19935 2012-01-25 09:06:14Z mol $

#ifndef LOFAR_IONPROC_CONTROL_PHASE3_CORES_H
#define LOFAR_IONPROC_CONTROL_PHASE3_CORES_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

//# Includes
#include <Interface/SmartPtr.h>
#include <Interface/Parset.h>
#include <Stream/Stream.h>
#include <Common/Thread/Semaphore.h>
#include <Common/Thread/Thread.h>

#include <string>
#include <vector>

namespace LOFAR {
namespace RTCP {

class ControlPhase3Cores {
  public:
				ControlPhase3Cores(const Parset &, const std::vector<Stream *> &phaseThreeStreams, unsigned firstBlock);
				~ControlPhase3Cores();

    void                        start();                            

    void			addIterations(unsigned count);
  
  private:
    void			mainLoop();

    const std::string		itsLogPrefix;

    const std::vector<Stream *>	&itsPhaseThreeStreams;
    const unsigned		itsMaxNrStreamsPerPset;
    const unsigned              itsFirstBlock;
    const bool                  itsAmNeeded;

    Semaphore			itsNrIterationsToDo;
    SmartPtr<Thread>		itsThread;
};

} // namespace RTCP
} // namespace LOFAR

#endif

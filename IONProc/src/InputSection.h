//#  InputSection.h: Catch RSP ethernet frames and synchronize RSP inputs 
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
//#  $Id: InputSection.h 17893 2011-04-29 09:04:10Z romein $

#ifndef LOFAR_IONPROC_INPUTSECTION_H
#define LOFAR_IONPROC_INPUTSECTION_H

// \file
// Catch RSP ethernet frames and synchronize RSP inputs 

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

//# Includes
#include <Interface/Parset.h>
#include <Interface/SmartPtr.h>
#include <Stream/Stream.h>
#include <BeamletBuffer.h>
#include <InputThread.h>
#include <LogThread.h>

#include <boost/multi_array.hpp>
#include <pthread.h>


namespace LOFAR {
namespace RTCP {

template <typename SAMPLE_TYPE> class InputSection
{
  public:
							InputSection(const Parset &, unsigned psetNumber);
							~InputSection();
  
    std::vector<SmartPtr<BeamletBuffer<SAMPLE_TYPE> > > itsBeamletBuffers;

  private:
    void						createInputStreams(const Parset &, const std::vector<Parset::StationRSPpair> &inputs);
    void						createInputThreads(const Parset &, const std::vector<Parset::StationRSPpair> &inputs);

    std::string						itsLogPrefix;

    std::vector<SmartPtr<Stream > >			itsInputStreams;
    
    unsigned						itsNrRSPboards;
   
    SmartPtr<LogThread>					itsLogThread;
    std::vector<SmartPtr<InputThread<SAMPLE_TYPE> > >	itsInputThreads;
};

} // namespace RTCP
} // namespace LOFAR

#endif

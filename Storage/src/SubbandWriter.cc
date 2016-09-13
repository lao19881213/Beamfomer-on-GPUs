//#  SubbandWriter.cc: Writes visibilities in an AIPS++ measurement set
//#
//#  Copyright (C) 2002-2005
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
//#  $Id: SubbandWriter.cc 22857 2012-11-20 08:58:52Z mol $

#include <lofar_config.h>

#include <Interface/DataFactory.h>
#include <Storage/SubbandWriter.h>


namespace LOFAR {
namespace RTCP {


SubbandWriter::SubbandWriter(const Parset &parset, OutputType outputType, unsigned streamNr, bool isBigEndian, const std::string &logPrefix)
{
  itsInputThread = new InputThread(parset, outputType, streamNr, itsFreeQueue, itsReceiveQueue, logPrefix);
  itsInputThread->start();

  try {
    itsOutputThread = new OutputThread(parset, outputType, streamNr, itsFreeQueue, itsReceiveQueue, logPrefix, isBigEndian);
    itsOutputThread->start();
  } catch (...) {
    itsInputThread->cancel();
    throw;
  }

  for (unsigned i = 0; i < maxReceiveQueueSize; i ++)
    itsFreeQueue.append(newStreamableData(parset, outputType, streamNr));
    
}

void SubbandWriter::augment( const FinalMetaData &finalMetaData )
{
  itsOutputThread->augment(finalMetaData);
}


} // namespace RTCP
} // namespace LOFAR

//#  SubbandWriter.h: Write subband(s) in an AIPS++ Measurement Set
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
//#  $Id: SubbandWriter.h 22857 2012-11-20 08:58:52Z mol $

#ifndef LOFAR_STORAGE_SUBBANDWRITER_H
#define LOFAR_STORAGE_SUBBANDWRITER_H

#include <Interface/OutputTypes.h>
#include <Interface/Parset.h>
#include <Interface/SmartPtr.h>
#include <Interface/StreamableData.h>
#include <Interface/FinalMetaData.h>
#include <Storage/InputThread.h>
#include <Storage/OutputThread.h>
#include <Common/Thread/Queue.h>

#include <string>


namespace LOFAR {
namespace RTCP {


class SubbandWriter
{
  public:
    SubbandWriter(const Parset &, OutputType, unsigned streamNr, bool isBigEndian, const std::string &logPrefix);

    void augment(const FinalMetaData &finalMetaData);

  private:
    static const unsigned	     maxReceiveQueueSize = 30;

    Queue<SmartPtr<StreamableData> > itsFreeQueue, itsReceiveQueue;

    SmartPtr<InputThread>	     itsInputThread;
    SmartPtr<OutputThread>	     itsOutputThread;
};


} // namespace RTCP
} // namespace LOFAR

#endif

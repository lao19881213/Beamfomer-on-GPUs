//# SubbandWriter.h: Write subband(s) in an AIPS++ Measurement Set
//# Copyright (C) 2008-2013  ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
//#
//# This file is part of the LOFAR software suite.
//# The LOFAR software suite is free software: you can redistribute it and/or
//# modify it under the terms of the GNU General Public License as published
//# by the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# The LOFAR software suite is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
//#
//# $Id: SubbandWriter.h 25646 2013-07-12 09:58:39Z mol $

#ifndef LOFAR_STORAGE_SUBBANDWRITER_H
#define LOFAR_STORAGE_SUBBANDWRITER_H

#include <string>

#include <Common/Thread/Queue.h>
#include <CoInterface/OutputTypes.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/StreamableData.h>
#include <CoInterface/FinalMetaData.h>
#include "InputThread.h"
#include "OutputThread.h"

namespace LOFAR
{
  namespace Cobalt
  {


    class SubbandWriter
    {
    public:
      SubbandWriter(const Parset &, OutputType, unsigned streamNr, bool isBigEndian, const std::string &logPrefix);

      void augment(const FinalMetaData &finalMetaData);

      ParameterSet feedbackLTA() const;

    private:
      static const unsigned maxReceiveQueueSize = 3;

      Queue<SmartPtr<StreamableData> > itsFreeQueue, itsReceiveQueue;

      SmartPtr<InputThread>            itsInputThread;
      SmartPtr<OutputThread>           itsOutputThread;
    };


  } // namespace Cobalt
} // namespace LOFAR

#endif


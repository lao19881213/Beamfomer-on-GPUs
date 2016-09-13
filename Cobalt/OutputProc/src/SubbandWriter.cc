//# SubbandWriter.cc: Writes visibilities in an AIPS++ measurement set
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
//# $Id: SubbandWriter.cc 25646 2013-07-12 09:58:39Z mol $

#include <lofar_config.h>

#include "SubbandWriter.h"

#include <CoInterface/DataFactory.h>

namespace LOFAR
{
  namespace Cobalt
  {


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

      for (unsigned i = 0; i < maxReceiveQueueSize; i++)
        itsFreeQueue.append(newStreamableData(parset, outputType, streamNr));

    }

    void SubbandWriter::augment( const FinalMetaData &finalMetaData )
    {
      itsOutputThread->augment(finalMetaData);
    }

    ParameterSet SubbandWriter::feedbackLTA() const
    {
      return itsOutputThread->feedbackLTA();
    }


  } // namespace Cobalt
} // namespace LOFAR


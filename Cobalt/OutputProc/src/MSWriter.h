//# MSMriter.h: Base class MSWriter
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
//# $Id: MSWriter.h 24385 2013-03-26 10:43:55Z amesfoort $

#ifndef LOFAR_STORAGE_MSWRITER_H
#define LOFAR_STORAGE_MSWRITER_H

#include <Common/ParameterSet.h>
#include <CoInterface/StreamableData.h>
#include <CoInterface/FinalMetaData.h>

namespace LOFAR
{
  namespace Cobalt
  {


    class MSWriter
    {
    public:
      MSWriter();
      virtual ~MSWriter();

      virtual void write(StreamableData *) = 0;

      virtual void augment(const FinalMetaData &finalMetaData);

      virtual size_t getDataSize();

      ParameterSet configuration() const;

      unsigned percentageWritten() const;

    protected:
      size_t itsNrBlocksWritten;
      size_t itsNrExpectedBlocks;
      ParameterSet itsConfiguration;
    };


  } // namespace Cobalt
} // namespace LOFAR

#endif


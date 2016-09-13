//# MSMriterCorrelated.h: a writer for correlated visibilities
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: MSWriterCorrelated.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_STORAGE_MSWRITERCORRELATED_H
#define LOFAR_STORAGE_MSWRITERCORRELATED_H

#include <string>
#include <vector>

#include <Stream/FileStream.h>
#include <CoInterface/Parset.h>
#include <CoInterface/SmartPtr.h>
#include <CoInterface/StreamableData.h>

#include "MSWriterFile.h"

namespace LOFAR
{
  namespace Cobalt
  {


    class MSWriterCorrelated : public MSWriterFile
    {
    public:
      MSWriterCorrelated(const std::string &logPrefix, const std::string &msName, const Parset &parset, unsigned subbandIndex, bool isBigEndian);
      ~MSWriterCorrelated();

      virtual void write(StreamableData *data);

      virtual void augment(const FinalMetaData &finalMetaData);

    protected:
      const std::string itsLogPrefix;
      const std::string itsMSname;
      const Parset &itsParset;

      SmartPtr<FileStream>             itsSequenceNumbersFile;

    private:
      void makeMeasurementSet(const std::string &logPrefix, const std::string &msName, const Parset &parset, unsigned subbandIndex, bool isBigEndian);
    };


  }
}

#endif


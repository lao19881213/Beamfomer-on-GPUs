//# MSMriterFile.h: raw file writer implementation of MSWriter
//# Copyright (C) 2009-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: MSWriterFile.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_STORAGE_MSWRITERFILE_H
#define LOFAR_STORAGE_MSWRITERFILE_H

#include <string>

#include <CoInterface/StreamableData.h>
#include "MSWriter.h"
#include "FastFileStream.h"

namespace LOFAR
{
  namespace Cobalt
  {


    class MSWriterFile : public MSWriter
    {
    public:
      MSWriterFile(const std::string &msName);
      ~MSWriterFile();

      virtual void write(StreamableData *data);

      virtual size_t getDataSize();

    protected:
      FastFileStream itsFile;
    };


  }
}

#endif


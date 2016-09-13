//# MSWriterFile.cc: a raw file writer
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
//# $Id: MSWriterFile.cc 24388 2013-03-26 11:14:29Z amesfoort $

#include <lofar_config.h>

#include "MSWriterFile.h"

#include <sys/types.h>
#include <fcntl.h>

namespace LOFAR
{
  namespace Cobalt
  {


    MSWriterFile::MSWriterFile (const std::string &msName)
      :
      itsFile(msName, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
    {
    }


    MSWriterFile::~MSWriterFile()
    {
    }


    void MSWriterFile::write(StreamableData *data)
    {
      data->write(&itsFile, true, FastFileStream::alignment);
    }


    size_t MSWriterFile::getDataSize()
    {
      return itsFile.size();
    }


  } // namespace Cobalt
} // namespace LOFAR


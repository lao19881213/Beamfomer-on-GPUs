//# FastFileStream.h: a FileStream using O_DIRECT
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
//# $Id: FastFileStream.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef LOFAR_STORAGE_FASTFILESTREAM_H
#define LOFAR_STORAGE_FASTFILESTREAM_H

#include <string>
#include <Stream/FileStream.h>
#include <CoInterface/SmartPtr.h>

namespace LOFAR
{
  namespace Cobalt
  {

    class FastFileStream : public FileStream
    {
    public:
      FastFileStream(const std::string &name, int flags, int mode); // rd/wr; create file

      virtual size_t tryWrite(const void *ptr, size_t size);
      virtual ~FastFileStream();

      virtual void skip( size_t bytes );

      virtual size_t size();

      // formally, the required alignment for O_DIRECT is determined by the file system
      static const unsigned alignment = 512;
    private:
      // writes the remainder, padded with zeros if needed. Returns the number of bytes written.
      size_t writeRemainder();

      // we only support writing
      virtual size_t tryRead(void *, size_t size)
      {
        return size;
      }

      // enlarge the buffer if needed
      void ensureBuffer(size_t newsize);

      // use the FileStream to force these data to disk
      void forceWrite(const void *ptr, size_t size);

      size_t bufsize;
      SmartPtr<char, SmartPtrFree<char> > buffer;
      size_t remainder;
    };

  }
}

#endif


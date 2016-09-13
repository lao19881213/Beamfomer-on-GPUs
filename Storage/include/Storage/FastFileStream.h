//  FastFileStream.h: a FileStream using O_DIRECT
//
//  Copyright (C) 2001
//  ASTRON (Netherlands Foundation for Research in Astronomy)
//  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//  $Id: MSWriterImpl.h 11891 2008-10-14 13:43:51Z gels $
//
//////////////////////////////////////////////////////////////////////


#ifndef LOFAR_STORAGE_FASTFILESTREAM_H
#define LOFAR_STORAGE_FASTFILESTREAM_H


#include <Stream/FileStream.h>
#include <Interface/SmartPtr.h>
#include <string>


namespace LOFAR {
namespace RTCP {

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
    virtual size_t tryRead(void *, size_t size) { return size; }

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

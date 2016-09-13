
//# FastFileStream.cc: a file writer using O_DIRECT
//#
//#  Copyright (C) 2001
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
//#  $Id: $

#include <lofar_config.h>

#include <Storage/FastFileStream.h>
#include <Interface/SmartPtr.h>
#include <Interface/Exceptions.h>
#include <Common/LofarLogger.h>
#include <Common/SystemCallException.h>

#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>

namespace LOFAR {
namespace RTCP {


FastFileStream::FastFileStream(const std::string &name, int flags, int mode)
:
  FileStream(name.c_str(), flags | O_DIRECT | O_SYNC, mode),
  bufsize(0),
  buffer(0),
  remainder(0)
{
  // alignment must be a power of two for easy calculations
  ASSERT( (alignment & (alignment-1)) == 0 );

  // alignment must be a multiple of sizeof(void*) for posix_memalign to work
  ASSERT( alignment % sizeof(void*) == 0 );
}

FastFileStream::~FastFileStream()
{
  // truncate the file to the exact right length
  try {
    errno = 0;

    off_t curlen = lseek(fd, 0, SEEK_CUR); // NOT SEEK_END, because skip() might push us beyond the end
    size_t origremainder = remainder;

    // lseek can return -1 as a valid file position, so check errno as well
    if (curlen == (off_t)-1 && errno)
      THROW_SYSCALL("lseek");

    writeRemainder();

    if (ftruncate(fd, curlen + origremainder) < 0)
      THROW_SYSCALL("ftruncate");
  } catch (Exception &ex) {
    LOG_ERROR_STR("Exception in destructor: " << ex);
  }

}

size_t FastFileStream::writeRemainder()
{
  if (remainder) {
    // pad with zeroes
    ensureBuffer(alignment);
    memset(buffer.get() + remainder, 0, alignment - remainder);
    forceWrite(buffer, alignment);

    remainder = 0;

    return alignment;
  }

  return 0;
}

void FastFileStream::ensureBuffer(size_t newsize)
{
  if (newsize <= bufsize)
    return;

  void *buf;

  if (posix_memalign(&buf, alignment, newsize) != 0)
    THROW( StorageException, "Not enough memory to allocate " << newsize << " bytes for fast writing");

  if (remainder) {
    ASSERT( buffer.get() );
    ASSERT( newsize >= remainder );

    memcpy(buf, buffer.get(), remainder);
  }

  buffer = static_cast<char*>(buf); // SmartPtr will take care of deleting the old buffer
  bufsize = newsize;
}     

void FastFileStream::forceWrite(const void *ptr, size_t size)
{
  // emulate Stream::write using FileStream::write to make sure all bytes are written
  while (size > 0) {
    ASSERT( (size & (alignment-1)) == 0 );
    ASSERT( (reinterpret_cast<size_t>(ptr) & (alignment-1)) == 0 );

    size_t bytes = FileStream::tryWrite(ptr, size);

    size -= bytes;
    ptr   = static_cast<const char *>(ptr) + bytes;
  }
}

size_t FastFileStream::tryWrite(const void *ptr, size_t size)
{
  const size_t orig_size = size;

  if (!remainder && (reinterpret_cast<size_t>(ptr) & (alignment-1)) == 0) {
    // pointer is aligned and we can write from it immediately

    ensureBuffer(alignment); // although remainder is enough, we want to avoid reallocating every time remainder grows slightly

    // save the remainder
    remainder = size & (alignment-1);
    memcpy(buffer.get(), static_cast<const char*>(ptr) + size - remainder, remainder);

    // write bulk
    forceWrite(ptr, size - remainder);
  } else {
    // not everything is aligned or there is a remainder -- use the buffer

    // move data to our buffer, and recompute new sizes
    ensureBuffer(alignment + size); // although remainder + size is enough, we want to avoid reallocating every time remainder grows slightly
    memcpy(buffer.get() + remainder, ptr, size);

    size += remainder;
    remainder = size & (alignment-1);

    // write bulk
    forceWrite(buffer.get(), size - remainder);

    // move remainder to the front
    memmove(buffer.get(), buffer.get() + size - remainder, remainder);
  }

  // lie about how many bytes we've written, since we might be caching
  // a remainder which we can't write to disk.
  return orig_size;
}


void FastFileStream::skip(size_t bytes)
{
  // make sure that the file pointer remains
  // at a full block boundary, so catch any
  // remainders.

  if (bytes == 0)
    return;

  // get rid of the old remainder first
  if (bytes + remainder >= alignment) {
    bytes -= (writeRemainder() - remainder);

    if (bytes >= alignment ) {
      // skip whole number of blocks
      size_t newremainder = bytes & (alignment - 1);
      size_t fullblocks = bytes - newremainder;

      FileStream::skip(fullblocks);

      bytes = newremainder;
    } 
  }  

  if (bytes > 0) {
    ASSERT( bytes < alignment );

    char zeros[bytes];

    tryWrite(&zeros, sizeof zeros);
  }  
}


size_t FastFileStream::size()
{
  // size we might have skip()ed and have some remaining data to write,
  // we cannot rely on FileStream::size(), which would report the current
  // file size, without skips or remainders in our buffer.

  errno = 0;

  off_t curlen = lseek(fd, 0, SEEK_CUR); // NOT SEEK_END, because skip() might push us beyond the end

  // lseek can return -1 as a valid file position, so check errno as well
  if (curlen == (off_t)-1 && errno)
    THROW_SYSCALL("lseek");

  return curlen + remainder;
}


} // namespace RTCP
} // namespace LOFAR


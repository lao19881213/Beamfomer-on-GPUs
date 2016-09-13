//# FCNP_ClientStream.cc: Fast Collective Network Protocol Stream
//#
//# Copyright (C) 2008
//# ASTRON (Netherlands Foundation for Research in Astronomy)
//# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//# This program is free software; you can redistribute it and/or modify
//# it under the terms of the GNU General Public License as published by
//# the Free Software Foundation; either version 2 of the License, or
//# (at your option) any later version.
//#
//# This program is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License
//# along with this program; if not, write to the Free Software
//# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//# $Id: FCNP_ClientStream.cc 15458 2010-04-15 15:32:36Z romein $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#if defined HAVE_FCNP && defined HAVE_BGP_CN

#include <Common/Timer.h>
#include <Interface/Align.h>
#include <Interface/AlignedStdAllocator.h>
#include <FCNP/fcnp_cn.h>
#include <FCNP_ClientStream.h>

#include <cstring>
#include <vector>


namespace LOFAR {
namespace RTCP {


FCNP_ClientStream::~FCNP_ClientStream()
{
}


size_t FCNP_ClientStream::tryRead(void *ptr, size_t size)
{
  //LOG_DEBUG_STR("FCNP_ClientStream::read(" << std::hex << ptr << ", " << std::dec << size << ", ...)");

  if (!aligned(ptr, 16) || !aligned(size, 16)) {
    size_t alignedSize = align(size, 16);
    std::vector<char, AlignedStdAllocator<char, 16> > alignedBuffer(alignedSize);

    FCNP_CN::IONtoCN_ZeroCopy(itsChannel, &alignedBuffer[0], alignedSize);
    memcpy(ptr, &alignedBuffer[0], size);
  } else {
    FCNP_CN::IONtoCN_ZeroCopy(itsChannel, ptr, size);
  }

  return size;
}


size_t FCNP_ClientStream::tryWrite(const void *ptr, size_t size)
{
  //LOG_DEBUG_STR("FCNP_ClientStream::write(" << std::hex << ptr << ", " << std::dec << size << ", ...)");

  if (!aligned(ptr, 16) || !aligned(size, 16)) {
    size_t alignedSize = align(size, 16);
    std::vector<char, AlignedStdAllocator<char, 16> > alignedBuffer(alignedSize);

    memcpy(&alignedBuffer[0], ptr, size);
    FCNP_CN::CNtoION_ZeroCopy(itsChannel, &alignedBuffer[0], alignedSize);
  } else {
    FCNP_CN::CNtoION_ZeroCopy(itsChannel, ptr, size);
  }

  return size;
}

} // namespace RTCP
} // namespace LOFAR

#endif

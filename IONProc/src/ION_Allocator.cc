//# ION_Allocator.cc: Class that allocates memory in the large-TLB area of the
//# I/O Node
//#
//# Copyright (C) 2006
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
//# $Id: ION_Allocator.cc 21689 2012-07-25 09:10:53Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <Interface/Allocator.h>
#include <ION_Allocator.h>

#include <cstdlib>
#include <iostream>

namespace LOFAR {
namespace RTCP {

#if defined FLAT_MEMORY
/*
 * Flat memory:
 *   Total size: 1.6 GByte
 *
 * Input section:
 *      4 * nrSlotsInFrame * nrSecondsOfBuffer * sampleRate * NR_POLARIZATIONS * sizeof(SAMPLE_TYPE) (BeamletBuffer.cc)
 *   <= 4 * 61             * 2.5               * 200e6/1024 * 2                * 2
 *    = 476562500 bytes < 455 MByte
 *
 * Output section:
 *    ~ 1.1 Gbyte left
 *    IONProc can output at most 1.1 GByte/s if it handles input (limited by CPU power)
 *    -> 1 second of buffer @ highest data rate
 *
 *    For lower data rates, we likely want to buffer more (f.e. CorrelatedData with an integration time of 1 second)
 */
static FixedArena  arena(reinterpret_cast<void *>(0x50000000), 0x60000000);
SparseSetAllocator hugeMemoryAllocator(arena);
#else
HeapAllocator	   &hugeMemoryAllocator = heapAllocator;
#endif


} // end namespace RTCP
} // end namespace LOFAR

//# PacketWriter.tcc
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: PacketWriter.tcc 25598 2013-07-08 12:31:36Z mol $

#include <ios>
#include <boost/format.hpp>

#include <Common/LofarTypes.h>
#include <Common/LofarConstants.h>
#include <Common/LofarLogger.h>
#include <InputProc/RSPTimeStamp.h>

namespace LOFAR {
namespace Cobalt {


template<typename T> PacketWriter<T>::PacketWriter( const std::string &logPrefix, SampleBuffer<T> &buffer, const struct BoardMode &mode, unsigned boardNr )
:
  logPrefix(str(boost::format("%s [PacketWriter] ") % logPrefix)),

  buffer(buffer),
  board(buffer.boards[boardNr]),
  mode(mode),
  settings(*buffer.settings),
  firstBeamlet(boardNr * mode.nrBeamletsPerBoard()),

  nrWritten(0)
{
  // Set mode for this board if necessary
  if (mode != *board.mode) {
    LOG_INFO_STR(logPrefix << "Switching buffer to mode " << mode.bitMode << " bit, " << mode.clockMHz << " MHz");

    board.changeMode(mode);
  }

  // bitmode must coincide with our template
  ASSERTSTR( T::bitMode() == mode.bitMode, "PacketWriter created with template for bitmode " << T::bitMode() << " but constructed for bitmode " << mode.bitMode );

  // our boardNr must exist
  ASSERT( boardNr < settings.nrBoards );
}

template<typename T>
void PacketWriter<T>::noMoreWriting()
{
  board.noMoreWriting();
}


template<typename T> void PacketWriter<T>::writePacket( const struct RSP &packet )
{
  if (packet.bitMode() != mode.bitMode || packet.clockMHz() != mode.clockMHz) {
    THROW(BadModeException, "Mode switch to " << packet.bitMode() << " bit, " << packet.clockMHz() << " MHz");
  }

  const uint8 &nrBeamlets  = packet.header.nrBeamlets;
  const uint8 &nrTimeslots = packet.header.nrBlocks;

  // should not exceed the number of beamlets we expect
  ASSERT( nrBeamlets <= mode.nrBeamletsPerBoard() );

  const TimeStamp begin = packet.timeStamp();
  const TimeStamp end   = begin + nrTimeslots;

  // determine the time span when cast on the buffer
  const size_t from_offset = buffer.offset(begin);
  size_t to_offset = buffer.offset(end);

  const size_t bufferSize = settings.nrSamples(T::bitMode());

  if (to_offset == 0)
    to_offset = bufferSize;

  const size_t wrap = from_offset < to_offset ? 0 : bufferSize - from_offset;

  /*
   * Make sure the buffer and available ranges are always consistent.
   */

  // signal write intent, to sync with reader in non-realtime mode and
  // to invalidate old data we're about to overwrite.
  board.startWrite(begin, end);

  // transpose
  const T *beamlets = reinterpret_cast<const T*>(&packet.payload.data);

  for (uint8 b = 0; b < nrBeamlets; ++b) {
    T *dst1 = &buffer.beamlets[firstBeamlet + b][from_offset];

    if (wrap > 0) {
      T *dst2 = &buffer.beamlets[firstBeamlet + b][0];

      memcpy(dst1, beamlets,        wrap        * sizeof(T));
      memcpy(dst2, beamlets + wrap, to_offset   * sizeof(T));
    } else {
      memcpy(dst1, beamlets, nrTimeslots * sizeof(T));
    }

    beamlets += nrTimeslots;
  }

  // mark as valid
  board.available.include(begin, end);

  // signal end of write
  board.stopWrite(end);

  ++nrWritten;
}


template<typename T> void PacketWriter<T>::logStatistics()
{
  LOG_INFO_STR( logPrefix << "Written " << nrWritten << " packets");

  nrWritten = 0;
}


}
}


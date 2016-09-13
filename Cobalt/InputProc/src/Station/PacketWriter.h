//# PacketWriter.h
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
//# $Id: PacketWriter.h 25540 2013-07-02 13:20:36Z mol $

#ifndef LOFAR_INPUT_PROC_PACKET_WRITER_H
#define LOFAR_INPUT_PROC_PACKET_WRITER_H

#include <cstring>
#include <string>

#include <InputProc/Buffer/SampleBuffer.h>
#include <InputProc/Buffer/BufferSettings.h>
#include <InputProc/Buffer/Ranges.h>

#include "RSP.h"

namespace LOFAR
{
  namespace Cobalt
  {
    EXCEPTION_CLASS(BadModeException, LOFAR::Exception);

    /*
     * Writes RSP packets to a SampleBuffer
     */
    template<typename T>
    class PacketWriter
    {
    public:
      PacketWriter( const std::string &logPrefix, SampleBuffer<T> &buffer, const struct BoardMode &mode, unsigned boardNr );

      // Signal end of (all) writing
      void noMoreWriting();

      // Write a packet to the SampleBuffer
      //
      // Throws BadModeException, if the packet was valid but did not correspond
      // to what the RSP board in `buffer' was configured for.
      void writePacket( const struct RSP &packet );

      void logStatistics();

    private:
      const std::string logPrefix;

      SampleBuffer<T> &buffer;
      typename SampleBuffer<T>::Board &board;

      // All packets must correspond to this mode, which we assume won't change
      // because each Board only sustains one writer (this object).
      const struct BoardMode mode;

      const struct BufferSettings &settings;
      const size_t firstBeamlet;

      size_t nrWritten;
    };


  }
}

#include "PacketWriter.tcc"

#endif


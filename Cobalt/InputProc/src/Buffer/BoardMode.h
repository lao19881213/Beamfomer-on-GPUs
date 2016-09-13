//# BoardMode.h
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
//# $Id: BoardMode.h 26336 2013-09-03 10:01:41Z mol $

#ifndef LOFAR_INPUT_PROC_BOARDMODE_H
#define LOFAR_INPUT_PROC_BOARDMODE_H

#include <Common/LofarTypes.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct BoardMode {
      unsigned bitMode;
      unsigned clockMHz;

      BoardMode(unsigned bitMode = 16, unsigned clockMHz = 200): bitMode(bitMode), clockMHz(clockMHz) {}

      // Returns the number of beamlets supported by a receiver for
      // an RSP board in this mode.
      unsigned nrBeamletsPerBoard() const {
        return nrBeamletsPerBoard(bitMode);
      }

      unsigned nrBeamletsPerBoard() const volatile {
        return nrBeamletsPerBoard(bitMode);
      }

      static unsigned nrBeamletsPerBoard( unsigned bitMode ) {
        // the number of beamlets scales with the bitmode:
        // 16-bit:  61
        //  8-bit: 122
        //  4-bit: 244
        switch (bitMode) {
          default:
          case 16:
            return nrBeamletsPerBoard_16bit;
          case 8:
            return nrBeamletsPerBoard_16bit << 1;
          case 4:
            return nrBeamletsPerBoard_16bit << 2;
        }
      }

      // Return the clock speed, in Hz
      unsigned clockHz() const {
        return clockMHz * 1000000;
      }

      // Convert a number of seconds to a number of samples
      int64 secondsToSamples(double seconds) const {
        return static_cast<int64>(seconds * clockHz() / 1024);
      }

      // Return the RSP board index corresponding to the provided beamlet
      size_t boardIndex(unsigned beamlet) const {
        return beamlet / nrBeamletsPerBoard();
      }

      /*
       * The functions below deal with comparing the volatile BoardMode as it
       * is stored in SHM with a non-volatile BoardMode as used in other code.
       */

      bool operator==(const volatile struct BoardMode &other) const {
        return bitMode == other.bitMode && clockMHz == other.clockMHz;
      }

      bool operator!=(const volatile struct BoardMode &other) const {
        return !(*this == other);
      }

      BoardMode(const volatile BoardMode &other): bitMode(other.bitMode), clockMHz(other.clockMHz) {}
      void operator=(const BoardMode &other) volatile { bitMode = other.bitMode; clockMHz = other.clockMHz; }

    private:
      // number of beamlets per RSP board in 16-bit mode.
      //
      // NOTE: this is actually the beamlet index increase between RSP boards.
      // Regardless of how many beamlets a packet actually carries, the second
      // RSP board starts sending from beamlet 61, leaving
      // a gap. For example, if each board sends 2 beamlets, then the beamlet
      // indices in the parset that can be used are:
      //
      // 0, 1, 61, 62, 122, 123, 183, 184.
      //
      // So it's best to leave nrBeamletsPerBoard_16bit at 61,
      // regardless of the number of beamlets contained in each packet.
      //
      // This value is hard-coded at the stations.
      static const unsigned nrBeamletsPerBoard_16bit = 61;
    };
  }
}

#endif


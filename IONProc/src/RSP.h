//#  RSP: RSP data format
//#
//#  Copyright (C) 2008
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
//#  $Id: RSP.h 23013 2012-11-28 09:03:30Z mol $

#ifndef LOFAR_IONPROC_RSP_H
#define LOFAR_IONPROC_RSP_H

#include <Common/LofarTypes.h>
#include <complex>

namespace LOFAR {
namespace RTCP {

#include <cstddef>


// WARNING: All data is in Little Endian format!
//
// Note that C++ bit fields are implementation dependent,
// so we cannot use them.

/* A structure fit for the maximum payload size. When reading UDP,
 * just read them straight into this struct, and ::read() will return
 * the size of the packet.
 *
 * When reading packets from file, make sure you read the right number
 * of bytes (see packetSize()).
 */

struct RSP {
  // ----------------------------------------------------------------------
  // Header and payload, in little endian!
  // ----------------------------------------------------------------------

  struct Header {
    // 2: Beamlet Data Interface 5.0
    // 3: Beamlet Data Interface 6.0 (8- and 4-bit mode support)
    uint8  version;

    // bit (0=LSB)
    //
    // 4:0    RSP board number
    // 5      (reserved, set to 0)
    // 6      0: payload ok, 1: payload has data errors
    // 7      0: 160 MHz     1: 200 MHz
    uint8  sourceInfo1;
    
    // bit (0=LSB)
    //
    // 1:0    0: 16-bit      1: 8-bit       2: 4-bit
    // 7:2    (reserved, set to 0)
    uint8  sourceInfo2;

    // identifiers
    uint8  configuration;
    uint16 station;

    // number of beamlets, typically at maximum:
    //   16-bit: 61
    //    8-bit: 122
    //    4-bit: 244
    uint8  nrBeamlets;

    // number of Xr+Xi+Yr+Yi samples per beamlet, typically 16
    uint8  nrBlocks;

    // UNIX timestamp in UTC (= # seconds since 1970-01-01 00:00:00)
    // 0xFFFFFFFF = clock not initialised
    uint32 timestamp;

    // Sample offset within the timestamp.
    //
    // 160 MHz: 160M/1024 = 156250 samples/second.
    //
    // 200 MHz: 200M/1024 = 195212.5 samples/second.
    //                      Even seconds have 195213 samples,
    //                      odd seconds have 195212 samples.
    uint32 blockSequenceNumber;
  } header;

  // Payload, allocated for maximum size.
  union {
    char data[8130];

    // samples are structured as samples[nrBlocks][nrBeamlets],
    // so first all blocks of the first beamlet, then all blocks of the second
    // beamlet, etc.
    //
    // for 4-bit mode:
    //  low octet: real      (2's complement)
    // high octet: imaginary (2's complement)

    struct { int16 Xr, Xi, Yr, Yi; } samples16bit[61 * 16];
    struct { int8  Xr, Xi, Yr, Yi; } samples8bit[122 * 16];
    struct { int8  X, Y;           } samples4bit[244 * 16];
  } payload;


  // ----------------------------------------------------------------------
  // Helper functions
  // ----------------------------------------------------------------------

  unsigned rspBoard() const {
    return header.sourceInfo1 & 0x1F;
  }

  bool payloadError() const {
    return header.sourceInfo1 & 0x40;
  }

  unsigned clockMHz() const {
    return header.sourceInfo1 & 0x80 ? 200 : 160;
  }

  unsigned bitMode() const {
    switch (header.sourceInfo2 & 0x3) {
      default:
      case 0x0: return 16;
      case 0x1: return 8;
      case 0x2: return 4;
    }
  }

  size_t packetSize() const {
    return sizeof(RSP::Header) + header.nrBlocks * header.nrBeamlets * 2 * 2 * bitMode() / 8;
  }


  // ----------------------------------------------------------------------
  // Payload decoding (for debug purposes, assumes data is converted to native
  // endianness)
  // ----------------------------------------------------------------------
  std::complex<int> sample( unsigned beamlet, unsigned block, char polarisation /* 'X' or 'Y' */) const {
    const unsigned offset = beamlet * header.nrBlocks + block;

    switch( bitMode() ) {
      default:
      case 16:
        return polarisation == 'X' ? std::complex<int>(payload.samples16bit[offset].Xr,
                                                       payload.samples16bit[offset].Xi)
                                   : std::complex<int>(payload.samples16bit[offset].Yr,
                                                       payload.samples16bit[offset].Yi);

      case 8:
        return polarisation == 'X' ? std::complex<int>(payload.samples8bit[offset].Xr,
                                                       payload.samples8bit[offset].Xi)
                                   : std::complex<int>(payload.samples8bit[offset].Yr,
                                                       payload.samples8bit[offset].Yi);

      case 4:
        return polarisation == 'X' ? decode4bit(payload.samples4bit[offset].X)
                                   : decode4bit(payload.samples4bit[offset].Y);
    }
  }

private:

  // decode the 4-bit complex type.
  static std::complex<int> decode4bit( int8 sample ) {
    int8 re = (sample << 4) >> 4; // preserve sign
    int8 im = (sample     ) >> 4; // preserve sign

    // balance range to [-7..7], subject to change!
    if (re == -8) re = -7;
    if (im == -8) im = -7;

    return std::complex<int>(re, im);
  }
};

} // namespace RTCP
} // namespace LOFAR

#endif

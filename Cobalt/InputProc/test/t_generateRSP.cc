//# t_generateRSP.cc
//#
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
//# $Id: t_generateRSP.cc 26881 2013-10-07 07:45:48Z loose $

#include <lofar_config.h>

#include <cmath>      // for min(), max()
#include <cstdlib>    // for rand(), srand(), system()
#include <fstream>
#include <iostream>

#include <boost/format.hpp>

#include <Common/LofarLogger.h>
#include <Stream/FileStream.h>
#include <InputProc/Station/PacketReader.h>

static const unsigned COMPLEX = 2;
static const unsigned NR_POLS = 2;
static const unsigned NR_BLOCKS = 16;

using namespace std;
using namespace boost;
using namespace LOFAR;
using namespace LOFAR::Cobalt;

void generate_input(ostream& os, unsigned bitMode,
                    unsigned nrPackets, unsigned nrSubbands)
{
  srand(0);
  const unsigned mask = (1 << bitMode) - 1;
  const unsigned offset = mask >> 1;
  for(unsigned packet = 0; packet < nrPackets; packet++) {
    for(unsigned subband = 0; subband < nrSubbands; subband++) {
      for(unsigned block = 0; block < NR_BLOCKS; block++) {
        for(unsigned pol = 0; pol < NR_POLS; pol++) {
          for(unsigned ri = 0; ri < COMPLEX; ri++) {
            int sample = min((rand() & mask) - offset, offset);
            os << sample << " ";
          }
        }
      }
      os << endl;
    }
    os << endl;
  }
}

void read_rsp(Stream& is, ostream& os, unsigned bitMode, unsigned nrSubbands)
{
  PacketReader reader("", is);
  RSP packet;
  complex<int> sample;
  try {
    while(true) {
      if (!reader.readPacket(packet)) continue;
      ASSERT(packet.bitMode() == bitMode);
      ASSERT(packet.header.nrBeamlets == nrSubbands);
      ASSERT(packet.header.nrBlocks == NR_BLOCKS);
      for(unsigned subband = 0; subband < nrSubbands; subband++) {
        for(unsigned block = 0; block < NR_BLOCKS; block++) {
          for(char pol = 'X'; pol <= 'Y'; pol++) {
            sample = packet.sample(subband, block, pol);
            os << sample.real() << " " << sample.imag() << " ";
          }
        }
        os << endl;
      }
      os << endl;
    }
  } catch (Stream::EndOfStreamException&) { }
}

int main()
{
  INIT_LOGGER("t_generateRSP");

  unsigned bitMode[] = {16, 8, 4};
  unsigned nrPackets[] = {1, 2, 3, 5};
  unsigned nrSubbands[] = {1, 37, 61, 103, 122, 156, 244, 250};
  string ascFile, rspFile, outFile, command;

  for(unsigned b = 0; b < sizeof(bitMode) / sizeof(unsigned); b++) {
    for(unsigned p = 0; p < sizeof(nrPackets) / sizeof(unsigned); p++) {
      for(unsigned s = 0; s < sizeof(nrSubbands) /sizeof(unsigned); s++) {

        cout << "bitmode: " << bitMode[b] << ", "
             << "nrPackets: " << nrPackets[p] << ", "
             << "nrSubbands: " << nrSubbands[s] << endl;

        // Skip if number of subbands exceeds capacity of one RSP board
        if(nrSubbands[s] > 122 * 8 / bitMode[b]) break;

        ascFile = str(format("t_generateRSP_tmp_b%d_p%d_s%d.asc") %
                      bitMode[b] % nrPackets[p] % nrSubbands[s]);
        rspFile = str(format("t_generateRSP_tmp_b%d_p%d_s%d.rsp") %
                      bitMode[b] % nrPackets[p] % nrSubbands[s]);
        outFile = str(format("t_generateRSP_tmp_b%d_p%d_s%d.out") %
                      bitMode[b] % nrPackets[p] % nrSubbands[s]);

        ofstream ascStream(ascFile.c_str());
        generate_input(ascStream, bitMode[b], nrPackets[p], nrSubbands[s]);

        command = str(format("../src/generateRSP -b%d -s%d < %s > %s") % 
                      bitMode[b] % nrSubbands[s] % ascFile % rspFile);
        cout << "Executing command: " << command << endl;
        ASSERT(system(command.c_str()) == 0);

        FileStream rspStream(rspFile);
        ofstream outStream(outFile.c_str());
        read_rsp(rspStream, outStream, bitMode[b], nrSubbands[s]);

        command = str(format("/usr/bin/diff -q %s %s") % ascFile % outFile);
        cout << "Executing command: " << command << endl;
        ASSERT(system(command.c_str()) == 0);
      }
    }
  }
}


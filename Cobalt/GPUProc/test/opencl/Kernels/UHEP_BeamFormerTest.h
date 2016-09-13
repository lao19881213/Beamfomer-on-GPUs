//# UHEP_BeamFormerTest.h
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
//# $Id: UHEP_BeamFormerTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_UHEP_BEAMFORMERTEST_H
#define GPUPROC_UHEP_BEAMFORMERTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/UHEP_BeamFormerKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct UHEP_BeamFormerTest : public UnitTest
    {
      UHEP_BeamFormerTest(const Parset &ps)
        :
        UnitTest(ps, "UHEP/BeamFormer.cl")
      {
        if (ps.nrStations() >= 5 && (ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1) >= 13 && ps.nrSubbands() >= 7 && ps.nrTABs(0) >= 6) {
          MultiArraySharedBuffer<char, 5> inputSamples(boost::extents[ps.nrStations()][ps.nrSubbands()][ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1][NR_POLARIZATIONS][ps.nrBytesPerComplexSample()], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 3> beamFormerWeights(boost::extents[ps.nrStations()][ps.nrSubbands()][ps.nrTABs(0)], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 4> complexVoltages(boost::extents[ps.nrSubbands()][ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1][ps.nrTABs(0)][NR_POLARIZATIONS], queue, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE);
          UHEP_BeamFormerKernel beamFormer(ps, program, complexVoltages, inputSamples, beamFormerWeights);

          switch (ps.nrBytesPerComplexSample()) {
          case 4: reinterpret_cast<std::complex<short> &>(inputSamples[4][6][12][1][0]) = std::complex<short>(2, 3);
            break;

          case 2: reinterpret_cast<std::complex<signed char> &>(inputSamples[4][6][12][1][0]) = std::complex<signed char>(2, 3);
            break;

          case 1: reinterpret_cast<i4complex &>(inputSamples[4][6][12][1][0]) = i4complex(2, 3);
            break;
          }

          beamFormerWeights[4][6][5] = std::complex<float>(4, 5);

          inputSamples.hostToDevice(CL_FALSE);
          beamFormerWeights.hostToDevice(CL_FALSE);
          beamFormer.enqueue(queue, counter);
          complexVoltages.deviceToHost(CL_TRUE);

          check(complexVoltages[6][12][5][1], std::complex<float>(-7, 22));
        }
      }
    };

  }
}

#endif


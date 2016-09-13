//# UHEP_TransposeTest.h
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
//# $Id: UHEP_TransposeTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_UHEP_TRANSPOSETEST_H
#define GPUPROC_UHEP_TRANSPOSETEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/UHEP_TransposeKernel.h>
#include <GPUProc/UHEP/InvertedStationPPFWeights.h> // reverseSubbandMapping

namespace LOFAR
{
  namespace Cobalt
  {
    struct UHEP_TransposeTest : public UnitTest
    {
      UHEP_TransposeTest(const Parset &ps)
        :
        UnitTest(ps, "UHEP/Transpose.cl")
      {
        if (ps.nrSubbands() >= 19 && ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1 >= 175 && ps.nrTABs(0) >= 5) {
          MultiArraySharedBuffer<std::complex<float>, 4> transposedData(boost::extents[ps.nrTABs(0)][NR_POLARIZATIONS][ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1][512], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 4> complexVoltages(boost::extents[ps.nrSubbands()][ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1][ps.nrTABs(0)][NR_POLARIZATIONS], queue, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY);
          cl::Buffer devReverseSubbandMapping(context, CL_MEM_READ_ONLY, 512 * sizeof(int));
          UHEP_TransposeKernel transpose(ps, program, transposedData, complexVoltages, devReverseSubbandMapping);

          complexVoltages[18][174][4][1] = std::complex<float>(24, 42);

          queue.enqueueWriteBuffer(devReverseSubbandMapping, CL_FALSE, 0, 512 * sizeof(int), reverseSubbandMapping);
          complexVoltages.hostToDevice(CL_FALSE);
          transpose.enqueue(queue, counter);
          transposedData.deviceToHost(CL_TRUE);

          check(transposedData[4][1][174][38], std::complex<float>(24, 42));
        }
      }
    };
  }
}

#endif


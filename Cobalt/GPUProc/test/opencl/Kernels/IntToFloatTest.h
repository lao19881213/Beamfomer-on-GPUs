//# IntToFloatTest.h
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
//# $Id: IntToFloatTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_INTTOFLOATTEST_H
#define GPUPROC_INTTOFLOATTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/IntToFloatKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct IntToFloatTest : public UnitTest
    {
      IntToFloatTest(const Parset &ps)
        :
        UnitTest(ps, "BeamFormer/IntToFloat.cl")
      {
        if (ps.nrStations() >= 3 && ps.nrSamplesPerChannel() * ps.nrChannelsPerSubband() >= 10077) {
          MultiArraySharedBuffer<char, 4> inputData(boost::extents[ps.nrStations()][ps.nrSamplesPerChannel() * ps.nrChannelsPerSubband()][NR_POLARIZATIONS][ps.nrBytesPerComplexSample()], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 3> outputData(boost::extents[ps.nrStations()][NR_POLARIZATIONS][ps.nrSamplesPerChannel() * ps.nrChannelsPerSubband()], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
          IntToFloatKernel kernel(ps, queue, program, outputData, inputData);

          switch (ps.nrBytesPerComplexSample()) {
          case 4: reinterpret_cast<std::complex<short> &>(inputData[2][10076][1][0]) = 7;
            break;

          case 2: reinterpret_cast<std::complex<signed char> &>(inputData[2][10076][1][0]) = 7;
            break;

          case 1: reinterpret_cast<i4complex &>(inputData[2][10076][1][0]) = i4complex(7, 0);
            break;
          }

          inputData.hostToDevice(CL_FALSE);
          kernel.enqueue(queue, counter);
          outputData.deviceToHost(CL_TRUE);
          check(outputData[2][1][10076], std::complex<float>(7.0f, 0));
        }
      }
    };

  }
}

#endif


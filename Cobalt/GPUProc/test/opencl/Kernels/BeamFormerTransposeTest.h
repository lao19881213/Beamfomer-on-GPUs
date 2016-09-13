//# BeamFormerTransposeTest.h
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
//# $Id: BeamFormerTransposeTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_BEAMFORMERTRANSPOSETEST_H
#define GPUPROC_BEAMFORMERTRANSPOSETEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/BeamFormerTransposeKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct BeamFormerTransposeTest : public UnitTest
    {
      BeamFormerTransposeTest(const Parset &ps)
        :
        UnitTest(ps, "BeamFormer/Transpose.cl")
      {
        if (ps.nrChannelsPerSubband() >= 19 && ps.nrSamplesPerChannel() >= 175 && ps.nrTABs(0) >= 5) {
          MultiArraySharedBuffer<std::complex<float>, 4> transposedData(boost::extents[ps.nrTABs(0)][NR_POLARIZATIONS][ps.nrSamplesPerChannel()][ps.nrChannelsPerSubband()], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 4> complexVoltages(boost::extents[ps.nrChannelsPerSubband()][ps.nrSamplesPerChannel()][ps.nrTABs(0)][NR_POLARIZATIONS], queue, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY);
          BeamFormerTransposeKernel transpose(ps, program, transposedData, complexVoltages);

          complexVoltages[18][174][4][1] = std::complex<float>(24, 42);

          complexVoltages.hostToDevice(CL_FALSE);
          transpose.enqueue(queue, counter);
          transposedData.deviceToHost(CL_TRUE);

          check(transposedData[4][1][174][18], std::complex<float>(24, 42));
        }
      }
    };
  }
}

#endif


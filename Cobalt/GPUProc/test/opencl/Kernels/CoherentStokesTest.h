//# CoherentStokesTest.h
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
//# $Id: CoherentStokesTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_COHERENTSTOKESTEST_H
#define GPUPROC_COHERENTSTOKESTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/CoherentStokesKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct CoherentStokesTest : public UnitTest
    {
      CoherentStokesTest(const Parset &ps)
        :
        UnitTest(ps, "BeamFormer/CoherentStokes.cl")
      {
        if (ps.nrChannelsPerSubband() >= 19 && ps.nrSamplesPerChannel() >= 175 && ps.nrTABs(0) >= 5) {
          MultiArraySharedBuffer<float, 4> stokesData(boost::extents[ps.nrTABs(0)][ps.nrCoherentStokes()][ps.nrSamplesPerChannel() / ps.coherentStokesTimeIntegrationFactor()][ps.nrChannelsPerSubband()], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
#if 1
          MultiArraySharedBuffer<std::complex<float>, 4> complexVoltages(boost::extents[ps.nrChannelsPerSubband()][ps.nrSamplesPerChannel()][ps.nrTABs(0)][NR_POLARIZATIONS], queue, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY);
          CoherentStokesKernel stokesKernel(ps, program, stokesData, complexVoltages);

          complexVoltages[18][174][4][0] = std::complex<float>(2, 3);
          complexVoltages[18][174][4][1] = std::complex<float>(4, 5);
#else
          MultiArraySharedBuffer<std::complex<float>, 4> complexVoltages(boost::extents[ps.nrTABs(0)][NR_POLARIZATIONS][ps.nrSamplesPerChannel()][ps.nrChannelsPerSubband()], queue, CL_MEM_READ_WRITE, CL_MEM_READ_ONLY);
          CoherentStokesKernel stokesKernel(ps, program, stokesData, complexVoltages);

          complexVoltages[18][174][4][0] = std::complex<float>(2, 3);
          complexVoltages[18][174][4][1] = std::complex<float>(4, 5);
#endif

          complexVoltages.hostToDevice(CL_FALSE);
          stokesKernel.enqueue(queue, counter);
          stokesData.deviceToHost(CL_TRUE);

          for (unsigned stokes = 0; stokes < ps.nrCoherentStokes(); stokes++)
            std::cout << stokesData[4][stokes][174 / ps.coherentStokesTimeIntegrationFactor()][18] << std::endl;
        }
      }
    };

  }
}

#endif


//# IncoherentStokesTest.h
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
//# $Id: IncoherentStokesTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_INCOHERENTSTOKESTEST_H
#define GPUPROC_INCOHERENTSTOKESTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/IncoherentStokesKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {

    struct IncoherentStokesTest : public UnitTest
    {
      IncoherentStokesTest(const Parset &ps)
        :
        UnitTest(ps, "BeamFormer/IncoherentStokes.cl")
      {
        if (ps.nrStations() >= 5 && ps.nrChannelsPerSubband() >= 14 && ps.nrSamplesPerChannel() >= 108) {
          MultiArraySharedBuffer<std::complex<float>, 4> inputData(boost::extents[ps.nrStations()][ps.nrChannelsPerSubband()][ps.nrSamplesPerChannel()][NR_POLARIZATIONS], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          MultiArraySharedBuffer<float, 3> stokesData(boost::extents[ps.nrIncoherentStokes()][ps.nrSamplesPerChannel() / ps.incoherentStokesTimeIntegrationFactor()][ps.nrChannelsPerSubband()], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
          IncoherentStokesKernel kernel(ps, queue, program, stokesData, inputData);

          inputData[4][13][107][0] = std::complex<float>(2, 3);
          inputData[4][13][107][1] = std::complex<float>(4, 5);

          inputData.hostToDevice(CL_FALSE);
          kernel.enqueue(queue, counter);
          stokesData.deviceToHost(CL_TRUE);

          const static float expected[] = { 54, -28, 46, 4 };

          for (unsigned stokes = 0; stokes < ps.nrIncoherentStokes(); stokes++)
            check(stokesData[stokes][107 / ps.incoherentStokesTimeIntegrationFactor()][13], expected[stokes]);
        }
      }
    };
  }
}

#endif


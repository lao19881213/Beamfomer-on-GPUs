//# CorrelatorTest.h
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
//# $Id: CorrelatorTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_CORRELATORTEST_H
#define GPUPROC_CORRELATORTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/CorrelatorKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct CorrelatorTest : public UnitTest
    {
      CorrelatorTest(const Parset &ps)
        :
#if defined USE_NEW_CORRELATOR
        UnitTest(ps, "NewCorrelator.cl")
#else
        UnitTest(ps, "Correlator.cl")
#endif
      {
        if (ps.nrStations() >= 5 && ps.nrChannelsPerSubband() >= 6 && ps.nrSamplesPerChannel() >= 100) {
          MultiArraySharedBuffer<std::complex<float>, 4> visibilities(boost::extents[ps.nrBaselines()][ps.nrChannelsPerSubband()][NR_POLARIZATIONS][NR_POLARIZATIONS], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 4> inputData(boost::extents[ps.nrStations()][ps.nrChannelsPerSubband()][ps.nrSamplesPerChannel()][NR_POLARIZATIONS], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          CorrelatorKernel correlator(ps, queue, program, visibilities, inputData);

          //inputData[3][5][99][1] = std::complex<float>(3, 4);
          //inputData[4][5][99][1] = std::complex<float>(5, 6);
          inputData[0][5][99][1] = std::complex<float>(3, 4);
          inputData[2][5][99][1] = std::complex<float>(5, 6);

          visibilities.hostToDevice(CL_FALSE);
          inputData.hostToDevice(CL_FALSE);
          correlator.enqueue(queue, counter);
          visibilities.deviceToHost(CL_TRUE);

          //check(visibilities[13][5][1][1], std::complex<float>(39, 2));
          //check(visibilities[5463][5][1][1], std::complex<float>(39, 2));
          for (unsigned bl = 0; bl < ps.nrBaselines(); bl++)
            if (visibilities[bl][5][1][1] != std::complex<float>(0, 0))
              std::cout << "bl = " << bl << ", visibility = " << visibilities[bl][5][1][1] << std::endl;
        }
      }
    };



  }
}

#endif


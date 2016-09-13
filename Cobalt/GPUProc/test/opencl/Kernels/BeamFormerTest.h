//# BeamFormerTest.h
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
//# $Id: BeamFormerTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_BEAMFORMERTEST_H
#define GPUPROC_BEAMFORMERTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/BeamFormerKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct BeamFormerTest : public UnitTest
    {
      BeamFormerTest(const Parset &ps)
        :
        UnitTest(ps, "BeamFormer/BeamFormer.cl")
      {
        if (ps.nrStations() >= 5 && ps.nrSamplesPerChannel() >= 13 && ps.nrChannelsPerSubband() >= 7 && ps.nrTABs(0) >= 6) {
          MultiArraySharedBuffer<std::complex<float>, 4> inputData(boost::extents[ps.nrStations()][ps.nrChannelsPerSubband()][ps.nrSamplesPerChannel()][NR_POLARIZATIONS], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 3> beamFormerWeights(boost::extents[ps.nrStations()][ps.nrChannelsPerSubband()][ps.nrTABs(0)], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          MultiArraySharedBuffer<std::complex<float>, 4> complexVoltages(boost::extents[ps.nrChannelsPerSubband()][ps.nrSamplesPerChannel()][ps.nrTABs(0)][NR_POLARIZATIONS], queue, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE);
          BeamFormerKernel beamFormer(ps, program, complexVoltages, inputData, beamFormerWeights);

          inputData[4][6][12][1] = std::complex<float>(2.2, 3);
          beamFormerWeights[4][6][5] = std::complex<float>(4, 5);

          inputData.hostToDevice(CL_FALSE);
          beamFormerWeights.hostToDevice(CL_FALSE);
          beamFormer.enqueue(queue, counter);
          complexVoltages.deviceToHost(CL_TRUE);

          check(complexVoltages[6][12][5][1], std::complex<float>(-6.2, 23));

#if 0
          for (unsigned tab = 0; tab < ps.nrTABs(0); tab++)
            for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol++)
              for (unsigned ch = 0; ch < ps.nrChannelsPerSubband(); ch++)
                for (unsigned t = 0; t < ps.nrSamplesPerChannel(); t++)
                  if (complexVoltages[tab][pol][ch][t] != std::complex<float>(0, 0))
                    std::cout << "complexVoltages[" << tab << "][" << pol << "][" << ch << "][" << t << "] = " << complexVoltages[tab][pol][ch][t] << std::endl;
#endif
        }
      }
    };
  }
}

#endif


//# DedispersionChirpTest.h
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
//# $Id: DedispersionChirpTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_DEDISPERSIONCHIRPTEST_H
#define GPUPROC_DEDISPERSIONCHIRPTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/DedispersionChirpKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {

    struct DedispersionChirpTest : public UnitTest
    {
      DedispersionChirpTest(const Parset &ps)
        :
        UnitTest(ps, "BeamFormer/Dedispersion.cl")
      {
        if (ps.nrTABs(0) > 3 && ps.nrChannelsPerSubband() > 13 && ps.nrSamplesPerChannel() / ps.dedispersionFFTsize() > 1 && ps.dedispersionFFTsize() > 77) {
          MultiArraySharedBuffer<std::complex<float>, 5> data(boost::extents[ps.nrTABs(0)][NR_POLARIZATIONS][ps.nrChannelsPerSubband()][ps.nrSamplesPerChannel() / ps.dedispersionFFTsize()][ps.dedispersionFFTsize()], queue, CL_MEM_READ_WRITE, CL_MEM_READ_WRITE);
          MultiArraySharedBuffer<float, 1> DMs(boost::extents[ps.nrTABs(0)], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
          DedispersionChirpKernel dedispersionChirpKernel(ps, program, queue, data, DMs);

          data[3][1][13][1][77] = std::complex<float>(2, 3);
          DMs[3] = 2;

          DMs.hostToDevice(CL_FALSE);
          data.hostToDevice(CL_FALSE);
          dedispersionChirpKernel.enqueue(queue, counter, 60e6);
          data.deviceToHost(CL_TRUE);

          std::cout << data[3][1][13][1][77] << std::endl;
        }
      }
    };
  }
}

#endif


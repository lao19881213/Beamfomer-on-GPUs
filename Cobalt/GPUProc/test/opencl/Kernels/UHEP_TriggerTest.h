//# UHEP_TriggerTest.h
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
//# $Id: UHEP_TriggerTest.h 24454 2013-03-28 13:37:14Z loose $

#ifndef GPUPROC_UHEP_TRIGGERTEST_H
#define GPUPROC_UHEP_TRIGGERTEST_H

#include <UnitTest.h>
#include <GPUProc/Kernels/UHEP_TriggerKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct UHEP_TriggerTest : public UnitTest
    {
      UHEP_TriggerTest(const Parset &ps)
        :
        UnitTest(ps, "UHEP/Trigger.cl")
      {
        if (ps.nrTABs(0) >= 4 && 1024 * ps.nrSamplesPerChannel() > 100015) {
          MultiArraySharedBuffer<float, 3> inputData(boost::extents[ps.nrTABs(0)][NR_POLARIZATIONS][ps.nrSamplesPerChannel() * 1024], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
          MultiArraySharedBuffer<TriggerInfo, 1> triggerInfo(boost::extents[ps.nrTABs(0)], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
          UHEP_TriggerKernel trigger(ps, program, triggerInfo, inputData);

          inputData[3][1][100015] = 1000;

          inputData.hostToDevice(CL_FALSE);
          trigger.enqueue(queue, counter);
          triggerInfo.deviceToHost(CL_TRUE);

          std::cout << "trigger info: mean = " << triggerInfo[3].mean << ", variance = " << triggerInfo[3].variance << ", bestValue = " << triggerInfo[3].bestValue << ", bestApproxIndex = " << triggerInfo[3].bestApproxIndex << std::endl;
          //check(triggerInfo[3].mean, (float) (1000.0f * 1000.0f) / (float) (ps.nrSamplesPerChannel() * 1024));
          check(triggerInfo[3].bestValue, 1000.0f * 1000.0f);
          check(triggerInfo[3].bestApproxIndex, 100016U);
        }
      }
    };

  }
}

#endif


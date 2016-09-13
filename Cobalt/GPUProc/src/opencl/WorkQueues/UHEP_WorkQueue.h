//# UHEP_SubbandProc.h
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
//# $Id: UHEP_WorkQueue.h 25727 2013-07-22 12:56:14Z mol $

#ifndef LOFAR_GPUPROC_OPENCL_UHEP_WORKQUEUE_H
#define LOFAR_GPUPROC_OPENCL_UHEP_WORKQUEUE_H

#include <complex>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/Pipelines/UHEP_Pipeline.h>
#include <GPUProc/Kernels/UHEP_TriggerKernel.h>
#include "SubbandProc.h"

namespace LOFAR
{
  namespace Cobalt
  {
    class UHEP_SubbandProc : public SubbandProc
    {
    public:
      UHEP_SubbandProc(UHEP_Pipeline &, unsigned queueNumber);

      void doWork(const float *delaysAtBegin, const float *delaysAfterEnd, const float *phaseOffsets);

      UHEP_Pipeline       &pipeline;
      cl::Event inputSamplesEvent, beamFormerWeightsEvent;

      cl::Buffer devBuffers[2];
      cl::Buffer devInputSamples;
      MultiArrayHostBuffer<char, 5> hostInputSamples;

      cl::Buffer devBeamFormerWeights;
      MultiArrayHostBuffer<std::complex<float>, 3> hostBeamFormerWeights;

      cl::Buffer devComplexVoltages;
      cl::Buffer devReverseSubbandMapping;
      cl::Buffer devFFTedData;
      cl::Buffer devInvFIRfilteredData;
      cl::Buffer devInvFIRfilterWeights;

      cl::Buffer devTriggerInfo;
      MultiArraySharedBuffer<TriggerInfo, 1> hostTriggerInfo;
    };
  }
}

#endif


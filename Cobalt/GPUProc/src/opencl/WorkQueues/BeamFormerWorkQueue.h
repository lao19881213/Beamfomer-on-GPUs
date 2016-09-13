//# BeamFormerSubbandProc.h
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
//# $Id: BeamFormerWorkQueue.h 25727 2013-07-22 12:56:14Z mol $

#ifndef LOFAR_GPUPROC_OPENCL_BEAM_FORMER_WORKQUEUE_H
#define LOFAR_GPUPROC_OPENCL_BEAM_FORMER_WORKQUEUE_H

#include <complex>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>

#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/BandPass.h>
#include <GPUProc/Pipelines/BeamFormerPipeline.h>

#include <GPUProc/Kernels/IntToFloatKernel.h>
#include <GPUProc/Kernels/Filter_FFT_Kernel.h>
#include <GPUProc/Kernels/DelayAndBandPassKernel.h>
#include <GPUProc/Kernels/BeamFormerKernel.h>
#include <GPUProc/Kernels/BeamFormerTransposeKernel.h>
#include <GPUProc/Kernels/DedispersionForwardFFTkernel.h>
#include <GPUProc/Kernels/DedispersionBackwardFFTkernel.h>
#include <GPUProc/Kernels/DedispersionChirpKernel.h>

#include "SubbandProc.h"

namespace LOFAR
{
  namespace Cobalt
  {
    class BeamFormerSubbandProc : public SubbandProc
    {
    public:
      BeamFormerSubbandProc(BeamFormerPipeline &, unsigned queueNumber);

      void doWork();

      BeamFormerPipeline  &pipeline;

      MultiArraySharedBuffer<char, 4>                inputSamples;
      DeviceBuffer devFilteredData;
      MultiArraySharedBuffer<float, 1>               bandPassCorrectionWeights;
      MultiArraySharedBuffer<float, 3>               delaysAtBegin, delaysAfterEnd;
      MultiArraySharedBuffer<float, 2>               phaseOffsets;
      DeviceBuffer devCorrectedData;
      MultiArraySharedBuffer<std::complex<float>, 3> beamFormerWeights;
      DeviceBuffer devComplexVoltages;
      MultiArraySharedBuffer<std::complex<float>, 4> transposedComplexVoltages;
      MultiArraySharedBuffer<float, 1>               DMs;

    private:
      IntToFloatKernel intToFloatKernel;
      Filter_FFT_Kernel fftKernel;
      DelayAndBandPassKernel delayAndBandPassKernel;
      BeamFormerKernel beamFormerKernel;
      BeamFormerTransposeKernel transposeKernel;
      DedispersionForwardFFTkernel dedispersionForwardFFTkernel;
      DedispersionBackwardFFTkernel dedispersionBackwardFFTkernel;
      DedispersionChirpKernel dedispersionChirpKernel;
    };

  }
}

#endif


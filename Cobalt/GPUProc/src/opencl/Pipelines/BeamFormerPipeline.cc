//# BeamFormerPipeline.cc
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
//# $Id: BeamFormerPipeline.cc 25727 2013-07-22 12:56:14Z mol $

#include <lofar_config.h>

#include "BeamFormerPipeline.h"

#include <Common/LofarLogger.h>

#include <GPUProc/OpenMP_Lock.h>
#include <GPUProc/SubbandProcs/BeamFormerSubbandProc.h>

namespace LOFAR
{
  namespace Cobalt
  {
    BeamFormerPipeline::BeamFormerPipeline(const Parset &ps)
      :
      Pipeline(ps),
      intToFloatCounter("int-to-float", profiling),
      fftCounter("FFT", profiling),
      delayAndBandPassCounter("delay/bp", profiling),
      beamFormerCounter("beamformer", profiling),
      transposeCounter("transpose", profiling),
      dedispersionForwardFFTcounter("ddisp.fw.FFT", profiling),
      dedispersionChirpCounter("chirp", profiling),
      dedispersionBackwardFFTcounter("ddisp.bw.FFT", profiling),
      samplesCounter("samples", profiling)
    {
      double startTime = omp_get_wtime();

#pragma omp parallel sections
      {
#pragma omp section
        intToFloatProgram = createProgram("BeamFormer/IntToFloat.cl");
#pragma omp section
        delayAndBandPassProgram = createProgram("DelayAndBandPass.cl");
#pragma omp section
        beamFormerProgram = createProgram("BeamFormer/BeamFormer.cl");
#pragma omp section
        transposeProgram = createProgram("BeamFormer/Transpose.cl");
#pragma omp section
        dedispersionChirpProgram = createProgram("BeamFormer/Dedispersion.cl");
      }

      LOG_DEBUG_STR("compile time = " << omp_get_wtime() - startTime);
    }

    void BeamFormerPipeline::doWork()
    {
#pragma omp parallel num_threads((profiling ? 1 : 2) * nrGPUs)
      BeamFormerSubbandProc(*this, omp_get_thread_num() % nrGPUs).doWork();
    }
  }
}


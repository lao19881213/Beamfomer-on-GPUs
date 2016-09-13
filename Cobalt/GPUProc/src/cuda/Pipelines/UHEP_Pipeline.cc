//# UHEP_Pipeline.cc
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
//# $Id: UHEP_Pipeline.cc 25727 2013-07-22 12:56:14Z mol $

#include <lofar_config.h>

#include "UHEP_Pipeline.h"

#include <Common/LofarLogger.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/OpenMP_Lock.h>
#include <GPUProc/SubbandProcs/UHEP_SubbandProc.h>

namespace LOFAR
{
  namespace Cobalt
  {
    UHEP_Pipeline::UHEP_Pipeline(const Parset &ps)
      :
      Pipeline(ps),
      beamFormerCounter("beamformer", profiling),
      transposeCounter("transpose", profiling),
      invFFTcounter("inv. FFT", profiling),
      invFIRfilterCounter("inv. FIR", profiling),
      triggerCounter("trigger", profiling),
      beamFormerWeightsCounter("BF weights", profiling),
      samplesCounter("samples", profiling)
    {
      double startTime = omp_get_wtime();

#pragma omp parallel sections
      {
#pragma omp section
        beamFormerProgram = createProgram("UHEP/BeamFormer.cl");
#pragma omp section
        transposeProgram = createProgram("UHEP/Transpose.cl");
#pragma omp section
        invFFTprogram = createProgram("UHEP/InvFFT.cl");
#pragma omp section
        invFIRfilterProgram = createProgram("UHEP/InvFIR.cl");
#pragma omp section
        triggerProgram = createProgram("UHEP/Trigger.cl");
      }

      LOG_DEBUG_STR("compile time = " << omp_get_wtime() - startTime);
    }

    void UHEP_Pipeline::doWork()
    {
      float delaysAtBegin[ps.nrBeams()][ps.nrStations()][NR_POLARIZATIONS] __attribute__((aligned(32)));
      float delaysAfterEnd[ps.nrBeams()][ps.nrStations()][NR_POLARIZATIONS] __attribute__((aligned(32)));
      float phaseOffsets[ps.nrStations()][NR_POLARIZATIONS] __attribute__((aligned(32)));

      memset(delaysAtBegin, 0, sizeof delaysAtBegin);
      memset(delaysAfterEnd, 0, sizeof delaysAfterEnd);
      memset(phaseOffsets, 0, sizeof phaseOffsets);
      delaysAtBegin[0][2][0] = 1e-6, delaysAfterEnd[0][2][0] = 1.1e-6;

#pragma omp parallel num_threads((profiling ? 1 : 2) * nrGPUs)
      UHEP_SubbandProc(*this, omp_get_thread_num() % nrGPUs).doWork(&delaysAtBegin[0][0][0], &delaysAfterEnd[0][0][0], &phaseOffsets[0][0]);
    }

  }
}


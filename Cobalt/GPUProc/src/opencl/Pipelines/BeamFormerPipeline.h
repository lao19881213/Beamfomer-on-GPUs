//# BeamFormerPipeline.h
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
//# $Id: BeamFormerPipeline.h 24840 2013-05-08 11:10:21Z loose $

#ifndef LOFAR_GPUPROC_OPENCL_BEAMFORMERPIPELINE_H
#define LOFAR_GPUPROC_OPENCL_BEAMFORMERPIPELINE_H

#include <CoInterface/Parset.h>

#include <GPUProc/gpu_incl.h>
#include <GPUProc/PerformanceCounter.h>
#include "Pipeline.h"

namespace LOFAR
{
  namespace Cobalt
  {
    class BeamFormerPipeline : public Pipeline
    {
    public:
      BeamFormerPipeline(const Parset &);

      void                    doWork();

      cl::Program intToFloatProgram, delayAndBandPassProgram, beamFormerProgram, transposeProgram, dedispersionChirpProgram;

      PerformanceCounter intToFloatCounter, fftCounter, delayAndBandPassCounter, beamFormerCounter, transposeCounter, dedispersionForwardFFTcounter, dedispersionChirpCounter, dedispersionBackwardFFTcounter;
      PerformanceCounter samplesCounter;
    };
  }
}

#endif


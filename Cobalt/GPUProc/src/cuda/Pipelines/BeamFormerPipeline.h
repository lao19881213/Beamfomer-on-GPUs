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
//# $Id: BeamFormerPipeline.h 26143 2013-08-21 12:36:16Z klijn $

#ifndef LOFAR_GPUPROC_CUDA_BEAMFORMERPIPELINE_H
#define LOFAR_GPUPROC_CUDA_BEAMFORMERPIPELINE_H

#include <vector>

#include <CoInterface/Parset.h>

#include "Pipeline.h"

namespace LOFAR
{
  namespace Cobalt
  {
    class BeamFormerPipeline : public Pipeline
    {
    public:
      BeamFormerPipeline(const Parset &, const std::vector<size_t> &subbandIndices, const std::vector<gpu::Device> &devices = gpu::Platform().devices());

      // When gpuProfiling isenabled will print the kernel statistics
      ~BeamFormerPipeline();
    };
  }
}

#endif


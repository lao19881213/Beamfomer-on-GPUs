//# FFT_Plan.h
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
//# $Id: FFT_Plan.h 26758 2013-09-30 11:47:21Z loose $

#ifndef LOFAR_GPUPROC_CUDA_FFT_PLAN_H
#define LOFAR_GPUPROC_CUDA_FFT_PLAN_H

#include <CoInterface/Parset.h>

#include <GPUProc/gpu_wrapper.h>
#include <cufft.h>

namespace LOFAR
{
  namespace Cobalt
  {

    class FFT_Plan
    {
    public:
      FFT_Plan(gpu::Context &context, unsigned fftSize, unsigned nrFFTs);
      ~FFT_Plan();

      // Tie the plan to a specific stream
      void setStream(const gpu::Stream &stream) const;

      cufftHandle plan;

    private:
      gpu::Context context;
    };
  }
}

#endif


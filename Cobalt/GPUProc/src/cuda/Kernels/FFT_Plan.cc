//# FFT_Plan.cc
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
//# $Id: FFT_Plan.cc 26991 2013-10-16 14:46:43Z mol $

#include <lofar_config.h>

#include "FFT_Plan.h"
#include <GPUProc/gpu_wrapper.h>

namespace LOFAR
{
  namespace Cobalt
  {
    FFT_Plan::FFT_Plan(gpu::Context &context, unsigned fftSize, unsigned nrFFTs)
      :
      context(context)
    {
      gpu::ScopedCurrentContext scc(context);

      cufftResult error;

      error = cufftPlan1d(&plan, fftSize, CUFFT_C2C, nrFFTs);

      if (error != CUFFT_SUCCESS)
        THROW(gpu::CUDAException, "cufftPlan1d: " << gpu::cufftErrorMessage(error));
    }

    FFT_Plan::~FFT_Plan()
    {
      gpu::ScopedCurrentContext scc(context);

      cufftDestroy(plan);
    }

    void FFT_Plan::setStream(const gpu::Stream &stream) const
    {
      cufftResult error;

      gpu::ScopedCurrentContext scc(context);

      error = cufftSetStream(plan, stream.get());

      if (error != CUFFT_SUCCESS)
        THROW(gpu::CUDAException, "cufftSetStream: " << gpu::cufftErrorMessage(error));
    }
  }
}


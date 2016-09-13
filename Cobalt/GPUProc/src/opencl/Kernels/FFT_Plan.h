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
//# $Id: FFT_Plan.h 24976 2013-05-21 13:42:00Z amesfoort $

#ifndef LOFAR_GPUPROC_OPENCL_FFT_PLAN_H
#define LOFAR_GPUPROC_OPENCL_FFT_PLAN_H

#include <CoInterface/Parset.h>

#include <GPUProc/gpu_incl.h>
#include <OpenCL_FFT/clFFT.h>

namespace LOFAR
{
  namespace Cobalt
  {

    class FFT_Plan
    {
    public:
      FFT_Plan(cl::Context &context, unsigned fftSize);
      ~FFT_Plan();
      clFFT_Plan plan;
    };
  }
}

#endif


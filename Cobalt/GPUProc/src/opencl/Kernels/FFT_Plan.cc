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
//# $Id: FFT_Plan.cc 24977 2013-05-21 13:46:16Z amesfoort $

#include <lofar_config.h>

#include "FFT_Plan.h"

namespace LOFAR
{
  namespace Cobalt
  {
    FFT_Plan::FFT_Plan(cl::Context &context, unsigned fftSize)
    {
      clFFT_Dim3 dim = { fftSize, 1, 1 };
      cl_int error;
      plan = clFFT_CreatePlan(context(), dim, clFFT_1D, clFFT_InterleavedComplexFormat, &error);

      if (error != CL_SUCCESS)
        throw cl::Error(error, "clFFT_CreatePlan");

      //clFFT_DumpPlan(plan, stdout);
    }

    FFT_Plan::~FFT_Plan()
    {
      clFFT_DestroyPlan(plan);
    }


  }
}


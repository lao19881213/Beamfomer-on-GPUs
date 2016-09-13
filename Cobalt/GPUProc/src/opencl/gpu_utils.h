//# gpu_utils.h
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: gpu_utils.h 24983 2013-05-21 16:10:38Z amesfoort $

#ifndef LOFAR_GPUPROC_OPENCL_GPU_UTILS_H
#define LOFAR_GPUPROC_OPENCL_GPU_UTILS_H

#include <string>
#include <vector>

#include <CoInterface/Parset.h>

#include "gpu_incl.h"

namespace LOFAR
{
  namespace Cobalt
  {

    void createContext(cl::Context &, std::vector<cl::Device> &);


    cl::Program createProgram(const Parset &ps, cl::Context &context,
                              std::vector<cl::Device> &devices,
                              const char *sources);
    // called by the above ("internal").
    cl::Program createProgram(cl::Context &, std::vector<cl::Device> &,
                              const char *sources, const char *args);

  } // namespace Cobalt
} // namespace LOFAR

#endif


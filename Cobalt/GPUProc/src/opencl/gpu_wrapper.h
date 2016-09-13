//# gpu_wrapper.h: OpenCL-specific wrapper classes for GPU types.
//#
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
//# $Id: gpu_wrapper.h 24983 2013-05-21 16:10:38Z amesfoort $

// \file opencl/gpu_wrapper.h
// GPU types on top of OpenCL.

#ifndef LOFAR_GPUPROC_OPENCL_GPU_WRAPPER_H
#define LOFAR_GPUPROC_OPENCL_GPU_WRAPPER_H

#include <sstream>

#include <Common/Exception.h>

#include <GPUProc/gpu_wrapper.h> // GPUException
#include "gpu_incl.h"

namespace LOFAR
{
  namespace Cobalt
  {
    namespace gpu 
    {
      // Exception class for GPU errors.
      EXCEPTION_CLASS(OpenCLException, GPUException);

      // Return the OpenCL error string associated with \a errcode.
      std::string errorMessage(cl_int errcode);

      // The sole purpose of this function is to extract detailed error
      // information if a cl::Error was thrown. Since we want the complete
      // backtrace, we cannot simply try-catch in main(), because that would
      // unwind the stack. The only option we have is to use our own terminate
      // handler.
      void terminate();
    }
  }
}

#if 0
//# Don't know how useful this is, because you can't wrap constructors this way.
//# If CL-exceptions are enabled the object will be defined inside the
//# try-block of the macro; if they're disabled the object will be defined 
//# inside the do-while block *and* you would need to check the error returned
//# in one of the constructor arguments :(

// Convenience macro to catch an OpenCL exception (cl::Error) and rethrow
// it as a LOFAR OpenCLException.
#if defined(__CL_ENABLE_EXCEPTIONS)
# define CHECK_OPENCL_CALL(func)                                        \
  try {                                                                 \
    func;                                                               \
  } catch (cl::Error &err) {                                            \
    std::ostringstream oss;                                             \
    oss << err.what() << ": " << LOFAR::Cobalt::gpu::errorMessage(err.err()); \
    THROW (LOFAR::Cobalt::gpu::OpenCLException, oss.str());             \
  }
#else
# define CHECK_OPENCL_CALL(func)                                        \
  do {                                                                  \
    cl_int result = func;                                               \
    if (result != CL_SUCCESS) {                                         \
      THROW (LOFAR::Cobalt::gpu::OpenCLException,                       \
             #func << ": " << LOFAR::Cobalt::gpu::errorMessage(result)); \
    }                                                                   \
  } while(0)
#endif
#endif

#endif


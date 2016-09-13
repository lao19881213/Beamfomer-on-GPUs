//# gpu_utils.h
//#
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
//# $Id: gpu_utils.h 26758 2013-09-30 11:47:21Z loose $

#ifndef LOFAR_GPUPROC_CUDA_GPU_UTILS_H
#define LOFAR_GPUPROC_CUDA_GPU_UTILS_H

#include <map>
#include <set>
#include <string>
#include <vector>

#include "gpu_wrapper.h"

namespace LOFAR
{
  namespace Cobalt
  {
    // Map for storing compile definitions that will be passed to the GPU kernel
    // compiler on the command line. The key is the name of the preprocessor
    // variable, a value should only be assigned if the preprocessor variable
    // needs to have a value.
    struct CompileDefinitions : std::map<std::string, std::string>
    {
    };

    // Return default compile definitions.
    CompileDefinitions defaultCompileDefinitions();

    // Set for storing compile flags that will be passed to the GPU kernel
    // compiler on the command line. Flags generally don't have an associated
    // value; if they do the value should become part of the flag in this set.
    struct CompileFlags : std::set<std::string>
    {
    };

    // Return default compile flags.
    CompileFlags defaultCompileFlags();

    // Vector for storing all the GPU devices present in the system.
    typedef std::vector<gpu::Device> GPUDevices;

    // Compile \a srcFilename and return the PTX code as string.
    // \par srcFilename Name of the file containing the source code to be
    //      compiled to PTX. If \a srcFilename is a relative path and if the
    //      environment variable \c LOFARROOT is set, then prefix \a srcFilename
    //      with the path \c $LOFARROOT/share/gpu/kernels.
    // \par flags Set of flags that need to be passed to the compiler on the
    //      command line.
    // \par definitions Map of key/value pairs containing the preprocessor
    //      variables that need to be passed to the compiler on the command
    //      line.
    // \par devices List of devices to compile for; the least capable device
    //      will be selected as target for the PTX code. If no devices are
    //      given, then the program is compiled for the least capable device
    //      that is available on the current platform.
    // \note The arguments \a srcFilename, \a flags and \a definitions are
    //       passed by value intentionally, because they will be modified by
    //       this method.
    std::string 
    createPTX(std::string srcFilename, 
              CompileDefinitions definitions = CompileDefinitions(),
              CompileFlags flags = CompileFlags(),
              const GPUDevices &devices = gpu::Platform().devices());

    // Create a Module from a PTX (string).
    // \par context The context that the Module should be associated with.
    // \par srcFilename Name of the
    gpu::Module createModule(const gpu::Context &context,
                             const std::string &srcFilename, 
                             const std::string &ptx);

    // Dump the contents of a device memory buffer as raw binary data to file.
    // \par deviceMemory Device memory buffer to be dumped.
    // \par dumpFile Name of the dump file.
    // \warning The underlying gpu::Stream must be synchronized, in order to
    // dump a device buffer. This may have a serious impact on performance.
    void dumpBuffer(const gpu::DeviceMemory &deviceMemory, 
                    const std::string &dumpFile);
  }
}

#endif


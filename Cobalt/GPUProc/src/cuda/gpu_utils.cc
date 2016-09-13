//# gpu_utils.cc
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
//# $Id: gpu_utils.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#include <GPUProc/gpu_utils.h>

#include <cstdlib>    // for getenv()
#include <cstdio>     // for popen(), pclose(), fgets()
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/format.hpp>

#include <Common/SystemCallException.h>
#include <Common/LofarLogger.h>
#include <CoInterface/Exceptions.h>

#include <GPUProc/global_defines.h>

#include "cuda_config.h"

namespace LOFAR
{
  namespace Cobalt
  {
    using namespace std;
    using boost::format;

    namespace {

      // Return the highest compute target supported by the given device
      CUjit_target computeTarget(const gpu::Device &device)
      {
        unsigned major = device.getComputeCapabilityMajor();
        unsigned minor = device.getComputeCapabilityMinor();

        switch (major) {
          case 0:
            return CU_TARGET_COMPUTE_10;

          case 1:
            switch (minor) {
              case 0:
                return CU_TARGET_COMPUTE_10;
              case 1:
                return CU_TARGET_COMPUTE_11;
              case 2:
                return CU_TARGET_COMPUTE_12;
              case 3:
                return CU_TARGET_COMPUTE_13;
              default:
                return CU_TARGET_COMPUTE_13;
            }

          case 2:
            switch (minor) {
              case 0:
                return CU_TARGET_COMPUTE_20;
              case 1:
                return CU_TARGET_COMPUTE_21;
              default:
                return CU_TARGET_COMPUTE_21;
            }

#if CUDA_VERSION >= 5000
          case 3:
            if (minor < 5) {
              return CU_TARGET_COMPUTE_30;
            } else {
              return CU_TARGET_COMPUTE_35;
            }

          default:
            return CU_TARGET_COMPUTE_35;
#else
          default:
            return CU_TARGET_COMPUTE_30;
#endif

        }
      }

      // Return the highest compute target supported by all the given devices
      CUjit_target computeTarget(const vector<gpu::Device> &devices)
      {
#if CUDA_VERSION >= 5000
        CUjit_target minTarget = CU_TARGET_COMPUTE_35;
#else
        CUjit_target minTarget = CU_TARGET_COMPUTE_30;
#endif

        for (vector<gpu::Device>::const_iterator i = devices.begin(); 
             i != devices.end(); ++i) {
          CUjit_target target = computeTarget(*i);

          if (target < minTarget)
            minTarget = target;
        }

        return minTarget;
      }

      // Translate a compute target to a virtual architecture (= the version
      // the .cu file is written in).
      string get_virtarch(CUjit_target target)
      {
        switch (target) {
        default:
          return "compute_unknown";

        case CU_TARGET_COMPUTE_10:
          return "compute_10";

        case CU_TARGET_COMPUTE_11:
          return "compute_11";

        case CU_TARGET_COMPUTE_12:
          return "compute_12";

        case CU_TARGET_COMPUTE_13:
          return "compute_13";

        case CU_TARGET_COMPUTE_20:
        case CU_TARGET_COMPUTE_21:
          // 21 not allowed for nvcc --gpu-architecture option value
          return "compute_20";

        case CU_TARGET_COMPUTE_30:
          return "compute_30";

#if CUDA_VERSION >= 5000
        case CU_TARGET_COMPUTE_35:
          return "compute_35";
#endif
        }
      }

      // Translate a compute target to a GPU architecture (= the instruction
      // set supported by the actual GPU).
      string get_gpuarch(CUjit_target target)
      {
        switch (target) {
        default:
          return "sm_unknown";

        case CU_TARGET_COMPUTE_10:
          return "sm_10";

        case CU_TARGET_COMPUTE_11:
          return "sm_11";

        case CU_TARGET_COMPUTE_12:
          return "sm_12";

        case CU_TARGET_COMPUTE_13:
          return "sm_13";

        case CU_TARGET_COMPUTE_20:
          return "sm_20";

        case CU_TARGET_COMPUTE_21:
          return "sm_21";

        case CU_TARGET_COMPUTE_30:
          return "sm_30";

#if CUDA_VERSION >= 5000
        case CU_TARGET_COMPUTE_35:
          return "sm_35";
#endif
        }
      }

      string lofarRoot()
      {
        // Prefer copy over racy static var or mutex.
        const char* env = getenv("LOFARROOT");
        return env ? string(env) : string();
      }

      string prefixPath()
      {
        if (lofarRoot().empty()) return ".";
        else return lofarRoot() + "/share/gpu/kernels";
      }

      string includePath()
      {
        if (lofarRoot().empty()) return "include";
        else return lofarRoot() + "/include";
      }

      ostream& operator<<(ostream& os, const CompileDefinitions& defs)
      {
        CompileDefinitions::const_iterator it;
        for (it = defs.begin(); it != defs.end(); ++it) {
          os << " -D" << it->first;
          if (!it->second.empty()) {
            os << "=" << it->second;
          }
        }
        return os;
      }

      ostream& operator<<(ostream& os, const CompileFlags& flags)
      {
        CompileFlags::const_iterator it;
        for (it = flags.begin(); it != flags.end(); ++it) {
          os << " " << *it;
        }
        return os;
      }

      string doCreatePTX(const string& source, 
                         const CompileFlags& flags,
                         const CompileDefinitions& defs)
      {
        // TODO: first try 'nvcc', then this path.
        ostringstream oss;
        oss << CUDA_TOOLKIT_ROOT_DIR << "/bin/nvcc " << source << flags << defs;
        string cmd(oss.str());
        LOG_DEBUG_STR("Starting runtime compilation:\n\t" << cmd);

        string ptx;
        char buffer [1024];       
        FILE * stream = popen(cmd.c_str(), "r");
        if (!stream) {
          THROW_SYSCALL("popen");
        }
        while (!feof(stream)) {  // NOTE: We do not get stderr (TODO)
          if (fgets(buffer, sizeof buffer, stream) != NULL) {
            ptx += buffer;
          }
        }
        if (pclose(stream) || ptx.empty()) {
          THROW(GPUProcException, "Runtime compilation failed!\n\t" << cmd);
        }
        return ptx;
      }

    } // namespace {anonymous}


    CompileDefinitions defaultCompileDefinitions()
    {
      CompileDefinitions defs;
      return defs;
    }

    CompileFlags defaultCompileFlags()
    {
      CompileFlags flags;
      flags.insert("-o /dev/stdout");
      flags.insert("-ptx");

      // For now, keep optimisations the same to detect changes in
      // output with reference.
      flags.insert("-use_fast_math");
      flags.insert("--restrict");
      flags.insert("-O3");

      flags.insert(str(format("-I%s") % includePath()));
      return flags;
    }

    string createPTX(string srcFilename, 
                     CompileDefinitions definitions,
                     CompileFlags flags, 
                     const vector<gpu::Device> &devices)
    {
      // The CUDA code is assumed to be written for the architecture of the
      // oldest device.
      flags.insert(str(format("--gpu-architecture %s") % 
                       get_virtarch(computeTarget(devices))));

      // Add default definitions and flags
      CompileDefinitions defaultDefinitions(defaultCompileDefinitions());
      definitions.insert(defaultDefinitions.begin(), 
                         defaultDefinitions.end());
      CompileFlags defaultFlags(defaultCompileFlags());
      flags.insert(defaultFlags.begin(),
                   defaultFlags.end());

#if 0
      // We'll compile a specific version for each device that has a different
      // architecture.
      set<CUjit_target> allTargets;

      for (vector<gpu::Device>::const_iterator i = devices.begin(); 
           i != devices.end(); ++i) {
        allTargets.add(computeTarget(*i));
      }

      for (set<CUjit_target>::const_iterator i = allTargets.begin();
           i != allTargets.end(); ++i) {
        flags.add(str(format("--gpu-code %s") % get_gpuarch(*i)));
      }
#endif

      // Prefix the CUDA kernel filename if it's a relative path.
      if (!srcFilename.empty() && srcFilename[0] != '/') {
        srcFilename = prefixPath() + "/" + srcFilename;
      }

      return doCreatePTX(srcFilename, flags, definitions);
    }


    gpu::Module createModule(const gpu::Context &context, 
                             const string &srcFilename,
                             const string &ptx)
    {
      const unsigned int BUILD_MAX_LOG_SIZE = 4095;
      /*
       * JIT compilation options.
       * Note: need to pass a void* with option vals. Preferably, do not alloc
       * dyn (mem leaks on exc).
       * Instead, use local vars for small variables and vector<char> xxx;
       * passing &xxx[0] for output c-strings.
       */
      gpu::Module::optionmap_t options;

#if 0
      size_t maxRegs = 63; // TODO: write this up
      options.push_back(CU_JIT_MAX_REGISTERS);
      optionValues.push_back(&maxRegs);

      size_t thrPerBlk = 256; // input and output val
      options.push_back(CU_JIT_THREADS_PER_BLOCK);
      optionValues.push_back(&thrPerBlk); // can be read back
#endif

      // input and output var for JIT compiler
      size_t infoLogSize  = BUILD_MAX_LOG_SIZE + 1;
      // idem (hence not the a single var or const)
      size_t errorLogSize = BUILD_MAX_LOG_SIZE + 1;

      vector<char> infoLog(infoLogSize);
      options[CU_JIT_INFO_LOG_BUFFER] = &infoLog[0];
      options[CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES] = 
        reinterpret_cast<void*>(infoLogSize);

      vector<char> errorLog(errorLogSize);
      options[CU_JIT_ERROR_LOG_BUFFER] = &errorLog[0];
      options[CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES] = 
        reinterpret_cast<void*>(errorLogSize);

      float &jitWallTime = reinterpret_cast<float&>(options[CU_JIT_WALL_TIME]);

#if 0
      size_t optLvl = 4; // 0-4, default 4
      options[CU_JIT_OPTIMIZATION_LEVEL] = reinterpret_cast<void*>(optLvl);
#endif

#if 0
      // NOTE: There is no need to specify a target. NVCC will use the best one
      // available based on the PTX and the Context.
      size_t jitTarget = target;
      options[CU_JIT_TARGET] = reinterpret_cast<void*>(jitTarget);
#endif

#if 0
      size_t fallback = CU_PREFER_PTX;
      options[CU_JIT_FALLBACK_STRATEGY] = reinterpret_cast<void*>(fallback);
#endif
      try {
        gpu::Module module(context, ptx.c_str(), options);
        // TODO: check what the ptx compiler prints. Don't print bogus. See if
        // infoLogSize indeed is set to 0 if all cool.
        // TODO: maybe retry if buffer len exhausted, esp for errors
        if (infoLogSize > infoLog.size()) {
          // zero-term log and guard against bogus JIT opt val output
          infoLogSize = infoLog.size();
        }
        infoLog[infoLogSize - 1] = '\0';
        LOG_DEBUG_STR( "Build info for '" << srcFilename 
             << "' (build time: " << jitWallTime 
             << " ms):" << endl << &infoLog[0] );

        return module;
      } catch (gpu::CUDAException& exc) {
        if (errorLogSize > errorLog.size()) { // idem
          errorLogSize = errorLog.size();
        }
        errorLog[errorLogSize - 1] = '\0';
        LOG_FATAL_STR( "Build errors for '" << srcFilename 
             << "' (build time: " << jitWallTime 
             << " ms):" << endl << &errorLog[0] );
        throw;
      }
    }

    void dumpBuffer(const gpu::DeviceMemory &deviceMemory, 
                    const std::string &dumpFile)
    {
      LOG_INFO_STR("Dumping device memory to file: " << dumpFile);
      gpu::HostMemory hostMemory(deviceMemory.fetch());
      std::ofstream ofs(dumpFile.c_str(), std::ios::binary);
      ofs.write(hostMemory.get<char>(), hostMemory.size());
    }

  } // namespace Cobalt
} // namespace LOFAR


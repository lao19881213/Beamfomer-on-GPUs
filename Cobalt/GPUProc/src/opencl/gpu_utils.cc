//# gpu_utils.cc
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
//# $Id: gpu_utils.cc 25094 2013-05-30 05:53:29Z amesfoort $

#include <lofar_config.h>

#include "gpu_utils.h"

#include <cstdlib>
#include <cstring>
#include <iomanip> // std::setprecision()
#include <fstream>
#include <sstream>

#include <Common/LofarLogger.h>
#include <Common/SystemUtil.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {

    void createContext(cl::Context &context, std::vector<cl::Device> &devices)
    {
      const char *platformName = getenv("PLATFORM");

#if defined __linux__
      if (platformName == 0)
#endif
      platformName = "NVIDIA CUDA";
      //platformName = "AMD Accelerated Parallel Processing";

      cl_device_type type = CL_DEVICE_TYPE_DEFAULT;

      const char *deviceType = getenv("TYPE");

      if (deviceType != 0) {
        if (strcmp(deviceType, "GPU") == 0)
          type = CL_DEVICE_TYPE_GPU;
        else if (strcmp(deviceType, "CPU") == 0)
          type = CL_DEVICE_TYPE_CPU;
        else
          LOG_ERROR_STR("Unrecognized device type: " << deviceType);
      }

      const char *deviceName = getenv("DEVICE");

      std::vector<cl::Platform> platforms;
      cl::Platform::get(&platforms);

      for (std::vector<cl::Platform>::iterator platform = platforms.begin(); platform != platforms.end(); platform++) {
        LOG_INFO_STR("Platform profile: " << platform->getInfo<CL_PLATFORM_PROFILE>());
        LOG_INFO_STR("Platform name: " << platform->getInfo<CL_PLATFORM_NAME>());
        LOG_INFO_STR("Platform version: " << platform->getInfo<CL_PLATFORM_VERSION>());
        LOG_INFO_STR("Platform extensions: " << platform->getInfo<CL_PLATFORM_EXTENSIONS>());
      }

      for (std::vector<cl::Platform>::iterator platform = platforms.begin(); platform != platforms.end(); platform++) {
        if (platform->getInfo<CL_PLATFORM_NAME>() == platformName) {
          platform->getDevices(type, &devices);

          if (deviceName != 0)
            for (std::vector<cl::Device>::iterator device = devices.end(); --device >= devices.begin(); )
              if (device->getInfo<CL_DEVICE_NAME>() != deviceName)
                devices.erase(device);

          for (std::vector<cl::Device>::iterator device = devices.begin(); device != devices.end(); device++) {
            LOG_INFO_STR("Device: " << device->getInfo<CL_DEVICE_NAME>());
            LOG_INFO_STR("Max mem: " << device->getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/1024/1024 << " MByte");
          }

          cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(*platform)(), 0 };
          context = cl::Context(type, cps);
          return;
        }
      }

      LOG_FATAL_STR("Platform not found: " << platformName);
      exit(1);
    }


    cl::Program createProgram(const Parset &ps, cl::Context &context, std::vector<cl::Device> &devices, const char *sources)
    {
      std::stringstream args;
      args << "-cl-fast-relaxed-math";

      std::vector<cl_context_properties> properties;
      context.getInfo(CL_CONTEXT_PROPERTIES, &properties);

      if (cl::Platform((cl_platform_id) properties[1]).getInfo<CL_PLATFORM_NAME>() == "NVIDIA CUDA") {
        args << " -cl-nv-verbose";
        args << " -cl-nv-opt-level=99";
        //args << " -cl-nv-maxrregcount=63";
        args << " -DNVIDIA_CUDA";
      }

      //if (devices[0].getInfo<CL_DEVICE_NAME>() == "GeForce GTX 680")
      //args << " -DUSE_FLOAT4_IN_CORRELATOR";

      args << " -I" << dirname(__FILE__);
      args << " -DNR_BITS_PER_SAMPLE=" << ps.nrBitsPerSample();
      args << " -DSUBBAND_BANDWIDTH=" << std::setprecision(7) << ps.subbandBandwidth() << 'f';
      args << " -DNR_SUBBANDS=" << ps.nrSubbands();
      args << " -DNR_CHANNELS=" << ps.nrChannelsPerSubband();
      args << " -DNR_STATIONS=" << ps.nrStations();
      args << " -DNR_SAMPLES_PER_CHANNEL=" << ps.nrSamplesPerChannel();
      args << " -DNR_SAMPLES_PER_SUBBAND=" << ps.nrSamplesPerSubband();
      args << " -DNR_BEAMS=" << ps.nrBeams();
      args << " -DNR_TABS=" << ps.nrTABs(0);
      args << " -DNR_COHERENT_STOKES=" << ps.nrCoherentStokes();
      args << " -DNR_INCOHERENT_STOKES=" << ps.nrIncoherentStokes();
      args << " -DCOHERENT_STOKES_TIME_INTEGRATION_FACTOR=" << ps.coherentStokesTimeIntegrationFactor();
      args << " -DINCOHERENT_STOKES_TIME_INTEGRATION_FACTOR=" << ps.incoherentStokesTimeIntegrationFactor();
      args << " -DNR_POLARIZATIONS=" << NR_POLARIZATIONS;
      args << " -DNR_TAPS=" << NR_TAPS;
      args << " -DNR_STATION_FILTER_TAPS=" << NR_STATION_FILTER_TAPS;

      if (ps.delayCompensation())
        args << " -DDELAY_COMPENSATION";

      if (ps.correctBandPass())
        args << " -DBANDPASS_CORRECTION";

      args << " -DDEDISPERSION_FFT_SIZE=" << ps.dedispersionFFTsize();
      return createProgram(context, devices, dirname(__FILE__).append("/").append(sources).c_str(), args.str().c_str());
    }

    cl::Program createProgram(cl::Context &context, std::vector<cl::Device> &devices, const char *sources, const char *args)
    {
      // Let the compiler read in the source file by passing an #include as source string.
      std::stringstream cmd;
      cmd << "#include \"" << std::string(sources) << '"' << std::endl;
      cl::Program program(context, cmd.str());

      try {
        program.build(devices, args);
        std::string msg;
        program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &msg);

        LOG_INFO(msg);
      } catch (cl::Error &error) {
        if (strcmp(error.what(), "clBuildProgram") == 0) {
          std::string msg;
          program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &msg);

          LOG_FATAL(msg);
          exit(1);
        } else {
          throw;
        }
      }

#if 1
      std::vector<size_t> binarySizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
#if 0
      // cl::Program::getInfo<> cl.hpp broken
      std::vector<char *> binaries = program.getInfo<CL_PROGRAM_BINARIES>();
#else
      std::vector<char *> binaries(binarySizes.size());

      for (unsigned b = 0; b < binaries.size(); b++)
        binaries[b] = new char[binarySizes[b]];

      cl_int error = clGetProgramInfo(program(), CL_PROGRAM_BINARIES, binaries.size() * sizeof(char *), &binaries[0], 0);

      if (error != CL_SUCCESS)
        throw cl::Error(error, "clGetProgramInfo");  // FIXME: cleanup binaries[*]
#endif

      for (unsigned i = 0; i < 1 /*binaries.size()*/; i++) {
        std::stringstream filename;
        filename << sources << '-' << i << ".ptx";
        std::ofstream(filename.str().c_str(), std::ofstream::binary).write(binaries[i], binarySizes[i]);
      }

#if 1
      for (unsigned b = 0; b < binaries.size(); b++)
        delete [] binaries[b];
#endif
#endif

      return program;
    }

  } // namespace Cobalt
} // namespace LOFAR


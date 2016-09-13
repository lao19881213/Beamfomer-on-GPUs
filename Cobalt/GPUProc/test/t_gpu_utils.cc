//# t_gpu_utils.cc : Unit tests for gpu_utils
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
//# $Id: t_gpu_utils.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#ifdef USE_CUDA

#include <cstdio>    // for remove()
#include <cstdlib>   // for unsetenv()
#include <vector>
#include <fstream>
#include <stdexcept>
#include <UnitTest++.h>
#include <Common/LofarLogger.h>
#include <CoInterface/Exceptions.h>
#include <GPUProc/gpu_utils.h>

using namespace std;
using namespace LOFAR::Cobalt;

#ifdef USE_CUDA
const char* srcFile("t_gpu_utils.cu");
#elif USE_OPENCL
const char* srcFile("t_gpu_utils.cl");
#else
#error "Either USE_CUDA or USE_OPENCL must be defined"
#endif

struct CreateFixture
{
  CreateFixture() {
    ofstream ofs(srcFile);
    if (!ofs) throw runtime_error("Failed to create file: " + string(srcFile));
    ofs << "#if defined FOO && FOO != 42\n"
        << "#error FOO != 42\n"
        << "#endif\n"
#ifdef USE_CUDA
        << "__global__ void dummy(void) {}\n"
#elif USE_OPENCL
        << "__kernel void dummy(__global void) {}\n"
#endif
        << endl;
  }
  ~CreateFixture() {
    remove(srcFile);
  }
};

TEST_FIXTURE(CreateFixture, CreatePtx)
{
  createPTX(srcFile);
}

TEST_FIXTURE(CreateFixture, CreatePtxExtraDef)
{
  CompileDefinitions defs;
  defs["FOO"] = "42";
  createPTX(srcFile, defs);
}

TEST_FIXTURE(CreateFixture, CreatePtxWrongExtraDef)
{
  CompileDefinitions defs;
  defs["FOO"] = "24";
  CHECK_THROW(createPTX(srcFile, defs), GPUProcException);
}

TEST_FIXTURE(CreateFixture, CreatePtxExtraFlag)
{
  CompileFlags flags;
  flags.insert("--source-in-ptx");
  createPTX(srcFile, defaultCompileDefinitions(), flags);
}

TEST_FIXTURE(CreateFixture, CreatePtxWrongExtraFlag)
{
  CompileFlags flags;
  flags.insert("--yaddayadda");
  CHECK_THROW(createPTX(srcFile, defaultCompileDefinitions(), flags), 
              GPUProcException);
}

TEST_FIXTURE(CreateFixture, CreateModule)
{
  gpu::Device device(gpu::Platform().devices()[0]);
  createModule(gpu::Context(device), srcFile, createPTX(srcFile));
}

TEST_FIXTURE(CreateFixture, CreateModuleHighestArch)
{
  // Highest known architecture is 3.5. 
  // Only perform this test if we do NOT have a device with that capability.
  gpu::Device device(gpu::Platform().devices()[0]);
  if (device.getComputeCapabilityMajor() == 3 && 
      device.getComputeCapabilityMinor() >= 5) return;
  CHECK_THROW(createModule(gpu::Context(device),
                           srcFile, 
                           createPTX(srcFile, 
                                     defaultCompileDefinitions(), 
                                     defaultCompileFlags(), 
                                     vector<gpu::Device>())),
              gpu::GPUException);
}

TEST(DumpBuffer)
{
  typedef unsigned element_type;
  const char* dumpFile("tDevMem.dat");
  const size_t num_elements(1024);
  const size_t num_bytes(num_elements * sizeof(element_type));
  vector<element_type> input(num_elements), output(num_elements);

  // Initialize input vector
  for(size_t i = 0; i < num_elements; i++) 
    input[i] = element_type(i);

  // Initialize GPU device, context, and stream.
  gpu::Device device(gpu::Platform().devices()[0]);
  gpu::Context ctx(device);
  gpu::Stream stream(ctx);

  // Allocate memory on host and device
  gpu::HostMemory hostMem(ctx, num_bytes);
  gpu::DeviceMemory devMem(ctx, num_bytes);

  // Copy input vector to host memory
  copy(input.begin(), input.end(), hostMem.get<element_type>());

  // Transfer to device memory
  stream.writeBuffer(devMem, hostMem);

  // Dump device memory to file
  dumpBuffer(devMem, dumpFile);

  // Read raw data back from file into output vector and remove file
  ifstream ifs(dumpFile, ios::binary);
  ifs.read(reinterpret_cast<char*>(&output[0]), num_bytes);
  remove(dumpFile);

  // Compare input and output vector.
  CHECK_ARRAY_EQUAL(input, output, num_elements);
}

int main()
{
  INIT_LOGGER("t_gpu_utils");

  unsetenv("LOFARROOT");

  try {
    gpu::Platform pf;
    return UnitTest::RunAllTests() > 0;
  } catch (gpu::GPUException& e) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 0;
  }
}

#else

#include <iostream>

int main()
{
  std::cout << "The GPU wrapper classes are not yet available for OpenCL.\n"
            << "Test skipped." << std::endl;
  return 0;
}

#endif

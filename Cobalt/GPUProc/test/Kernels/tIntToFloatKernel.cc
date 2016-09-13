//# tDelayAndBandPassKernel.cc: test Kernels/DelayAndBandPassKernel class
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
//# $Id: tDelayAndBandPassKernel.cc 25199 2013-06-05 23:46:56Z amesfoort $

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <CoInterface/BlockID.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/BandPass.h>
#include <GPUProc/Kernels/IntToFloatKernel.h>
#include <GPUProc/SubbandProcs/CorrelatorSubbandProc.h>

using namespace std;
using namespace LOFAR::Cobalt;

int main() {
  INIT_LOGGER("tIntToFloatKernel");
  
  
  // Set up gpu environment
  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " GPU devices" << endl;
  } catch (gpu::GPUException& e) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 3;
  }
  gpu::Device device(0);
  vector<gpu::Device> devices(1, device);
  gpu::Context ctx(device);
  gpu::Stream stream(ctx);

  Parset ps("tIntToFloatKernel.in_parset");
  KernelFactory<IntToFloatKernel> factory(ps);

  size_t nSampledData = factory.bufferSize(IntToFloatKernel::INPUT_DATA) / sizeof(char);
  size_t sizeSampledData = nSampledData * sizeof(char);

  // Create some initialized host data
  gpu::HostMemory sampledData(ctx, sizeSampledData);
  char *samples = sampledData.get<char>();
  for (unsigned idx =0; idx < nSampledData; ++idx)
    samples[idx] = -128;  // set all to -128
  gpu::DeviceMemory devSampledData(ctx, factory.bufferSize(IntToFloatKernel::INPUT_DATA));
  stream.writeBuffer(devSampledData, sampledData, true);
  
  // Device mem for output
  gpu::DeviceMemory devConvertedData(ctx, factory.bufferSize(IntToFloatKernel::OUTPUT_DATA));
  gpu::HostMemory convertedData(ctx,  factory.bufferSize(IntToFloatKernel::OUTPUT_DATA));
  //stream.writeBuffer(devConvertedData, sampledData, true);

  IntToFloatKernel::Buffers buffers(devSampledData, devConvertedData);
  auto_ptr<IntToFloatKernel> kernel(factory.create(stream, buffers));

  BlockID blockId;
  kernel->enqueue(blockId);
  stream.synchronize();
  stream.readBuffer(convertedData, devConvertedData, true);
  stream.synchronize();
  float *samplesFloat = convertedData.get<float>();
  
  // Validate the output:
  // The inputs were all -128 with bits per sample 8. 
  // Therefore they should all be converted to -127 (but scaled to 16 bit amplitute values).
  for (size_t idx =0; idx < nSampledData; ++idx)
    if (samplesFloat[idx] != -127 * 16)
    {
        cerr << "Found an uncorrect sample in the output array at idx: " << idx << endl
             << "Value found: " << samplesFloat[idx] << endl
             << "Test failed "  << endl;
        return 1;
    }
      
  return 0;
}


//# tBandPassCorrectionKernel.cc: test Kernels/BandPassCorrectionKernel class
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
//# $Id: tBandPassCorrectionKernel.cc 27265 2013-11-06 13:28:18Z klijn $

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/BandPass.h>
#include <GPUProc/Kernels/BandPassCorrectionKernel.h>
#include <GPUProc/SubbandProcs/CorrelatorSubbandProc.h>
#include <GPUProc/PerformanceCounter.h>
#include <CoInterface/BlockID.h>

using namespace std;
using namespace LOFAR::Cobalt;

int main() {
  INIT_LOGGER("tBandPassCorrectionKernel");

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

  Parset ps("tBandPassCorrectionKernel.in_parset");
  BandPassCorrectionKernel::Parameters params(ps);
  params.nrChannels1 = 64;
  params.nrChannels2 = 64;
  params.nrSamplesPerChannel = 
          ps.nrSamplesPerSubband() / (params.nrChannels1 * params.nrChannels2);

  KernelFactory<BandPassCorrectionKernel> factory(params);

  // Get the buffers as created by factory
  gpu::DeviceMemory 
    inputData(ctx, factory.bufferSize(BandPassCorrectionKernel::INPUT_DATA)),
    filteredData(ctx, factory.bufferSize(BandPassCorrectionKernel::OUTPUT_DATA)),
    bandPassCorrectionWeights(ctx, factory.bufferSize(BandPassCorrectionKernel::BAND_PASS_CORRECTION_WEIGHTS));

  BandPassCorrectionKernel::Buffers buffers(inputData, filteredData, bandPassCorrectionWeights);

  auto_ptr<BandPassCorrectionKernel> kernel(factory.create(stream, buffers));

  PerformanceCounter counter(ctx);
  BlockID blockId;
  kernel->enqueue(blockId, counter);
  stream.synchronize();

  return 0;
}


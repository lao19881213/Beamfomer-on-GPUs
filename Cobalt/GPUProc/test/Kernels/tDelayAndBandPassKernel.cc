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
//# $Id: tDelayAndBandPassKernel.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/BandPass.h>
#include <GPUProc/Kernels/DelayAndBandPassKernel.h>
#include <GPUProc/SubbandProcs/CorrelatorSubbandProc.h>
#include <GPUProc/PerformanceCounter.h>
#include <CoInterface/BlockID.h>

using namespace std;
using namespace LOFAR::Cobalt;

int main() {
  INIT_LOGGER("tDelayAndBandPassKernel");

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

  Parset ps("tDelayAndBandPassKernel.in_parset");
  KernelFactory<DelayAndBandPassKernel> factory(ps);

  gpu::DeviceMemory 
    inputData(ctx, factory.bufferSize(DelayAndBandPassKernel::INPUT_DATA)),
    filteredData(ctx, factory.bufferSize(DelayAndBandPassKernel::OUTPUT_DATA)),
    delaysAtBegin(ctx, factory.bufferSize(DelayAndBandPassKernel::DELAYS)),
    delaysAfterEnd(ctx, factory.bufferSize(DelayAndBandPassKernel::DELAYS)),
    phaseOffsets(ctx, factory.bufferSize(DelayAndBandPassKernel::PHASE_OFFSETS)),
    bandPassCorrectionWeights(ctx, factory.bufferSize(DelayAndBandPassKernel::BAND_PASS_CORRECTION_WEIGHTS));

  DelayAndBandPassKernel::Buffers buffers(inputData, filteredData, delaysAtBegin, delaysAfterEnd, phaseOffsets, bandPassCorrectionWeights);

  auto_ptr<DelayAndBandPassKernel> kernel(factory.create(stream, buffers));

  size_t subbandIdx = 0;
  float centralFrequency = ps.settings.subbands[subbandIdx].centralFrequency;
  size_t SAP = ps.settings.subbands[subbandIdx].SAP;
  PerformanceCounter counter(ctx);
  BlockID blockId;
  kernel->enqueue(blockId, counter, centralFrequency, SAP);
  stream.synchronize();

  return 0;
}


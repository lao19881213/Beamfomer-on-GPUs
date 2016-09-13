//# tBeamFormerKernel.cc: test Kernels/BeamFormerKernel class
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
//# $Id: tBeamFormerKernel.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/BandPass.h>
#include <GPUProc/Kernels/BeamFormerKernel.h>
#include <CoInterface/BlockID.h>
#include "../TestUtil.h"

#include <boost/lexical_cast.hpp>
#include <GPUProc/PerformanceCounter.h>

using namespace std;
using namespace boost;
using namespace LOFAR::Cobalt::gpu;
using namespace LOFAR::Cobalt;

int main() {
  INIT_LOGGER("tBeamFormerKernel");

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

  Parset ps("tBeamFormerKernel.in_parset");

  KernelFactory<BeamFormerKernel> factory(ps);

  // Calculate beamformer delays and transfer to the device.
  // *************************************************************
  size_t lengthDelaysData = 
    factory.bufferSize(BeamFormerKernel::BEAM_FORMER_DELAYS) / 
    sizeof(float);
  size_t lengthBandPassCorrectedData =
    factory.bufferSize(BeamFormerKernel::INPUT_DATA) /
    sizeof(float);
  size_t lengthComplexVoltagesData = 
    factory.bufferSize(BeamFormerKernel::OUTPUT_DATA) / 
    sizeof(float);

  // Define the input and output arrays
  // ************************************************************* 
  float * complexVoltagesData = new float[lengthComplexVoltagesData];
  float * bandPassCorrectedData = new float[lengthBandPassCorrectedData];
  float * delaysData= new float[lengthDelaysData];

  // ***********************************************************
  // Baseline test: If all weight data is zero the output should be zero
  // The output array is initialized with 42s
  cout << "test 1" << endl;
  for (unsigned idx = 0; idx < lengthComplexVoltagesData / 2; ++idx)
  {
    complexVoltagesData[idx * 2] = 42.0f;
    complexVoltagesData[idx * 2 + 1] = 42.0f;
  }

  for (unsigned idx = 0; idx < lengthBandPassCorrectedData / 2; ++idx)
  {
    bandPassCorrectedData[idx * 2] = 1.0f;
    bandPassCorrectedData[idx * 2+ 1] = 1.0f;
  }

  for (unsigned idx = 0; idx < lengthDelaysData/2; ++idx)
  {
    delaysData[idx * 2] = 0.0f;
    delaysData[idx * 2 + 1] = 0.0f;
  }

  size_t sizeDelaysData = lengthDelaysData * sizeof(float);
  DeviceMemory devDelaysMemory(ctx, sizeDelaysData);
  HostMemory rawDelaysData = getInitializedArray(ctx, sizeDelaysData, 1.0f);
  float *rawDelaysPtr = rawDelaysData.get<float>();
  for (unsigned idx = 0; idx < lengthDelaysData; ++idx)
    rawDelaysPtr[idx] = delaysData[idx];
  stream.writeBuffer(devDelaysMemory, rawDelaysData);

  size_t sizeBandPassCorrectedData = 
    lengthBandPassCorrectedData * sizeof(float);
  DeviceMemory devBandPassCorrectedMemory(ctx, sizeBandPassCorrectedData);
  HostMemory rawBandPassCorrectedData = 
    getInitializedArray(ctx, sizeBandPassCorrectedData, 2.0f);
  float *rawBandPassCorrectedPtr = rawBandPassCorrectedData.get<float>();
  for (unsigned idx = 0; idx < lengthBandPassCorrectedData; ++idx)
    rawBandPassCorrectedPtr[idx] = bandPassCorrectedData[idx];
  stream.writeBuffer(devBandPassCorrectedMemory, rawBandPassCorrectedData);

  size_t sizeComplexVoltagesData = lengthComplexVoltagesData * sizeof(float);
  DeviceMemory devComplexVoltagesMemory(ctx, sizeComplexVoltagesData);
  HostMemory rawComplexVoltagesData = 
    getInitializedArray(ctx, sizeComplexVoltagesData, 3.0f);
  float *rawComplexVoltagesPtr = rawComplexVoltagesData.get<float>();
  for (unsigned idx = 0; idx < lengthComplexVoltagesData; ++idx)
    rawComplexVoltagesPtr[idx] = complexVoltagesData[idx];

  // Write output content.
  stream.writeBuffer(devComplexVoltagesMemory, rawComplexVoltagesData);

  BeamFormerKernel::Buffers buffers(devBandPassCorrectedMemory, devComplexVoltagesMemory, devDelaysMemory);

  auto_ptr<BeamFormerKernel> kernel(factory.create(stream, buffers));

  float subbandFreq = 60e6f;
  unsigned sap = 0;

  PerformanceCounter counter(ctx);
  BlockID blockId;
  kernel->enqueue(blockId, counter, subbandFreq, sap);
  stream.synchronize();

  return 0;
}


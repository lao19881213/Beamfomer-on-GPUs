//# tTranspose.cc: test Transpose CUDA kernel
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
//# $Id: tTranspose.cc 25978 2013-08-08 03:26:50Z amesfoort $

#include <lofar_config.h>

#include <cstdlib>  // for rand()
#include <string>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <Common/Exception.h>
#include <Common/LofarLogger.h>

#include <CoInterface/BlockID.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/Kernels/BeamFormerTransposeKernel.h>

using namespace std;
using namespace boost;
using namespace LOFAR::Cobalt::gpu;
using namespace LOFAR::Cobalt;
using namespace LOFAR::TYPES;
using LOFAR::Exception;

unsigned NR_CHANNELS = 2048;
unsigned NR_SAMPLES_PER_CHANNEL = 16;
unsigned NR_TABS = 2;
unsigned NR_POLARIZATIONS = 2;

Exception::TerminateHandler t(Exception::terminate);

void runTest( Context &ctx, Stream &stream )
{
  Parset ps;
  ps.add("Observation.DataProducts.Output_Beamformed.enabled", "true");
  ps.updateSettings();

  BeamFormerTransposeKernel::Parameters params(ps);
  params.nrChannelsPerSubband = NR_CHANNELS;
  params.nrSamplesPerChannel = NR_SAMPLES_PER_CHANNEL;
  params.nrTABs = NR_TABS;

  KernelFactory<BeamFormerTransposeKernel> factory(params);

  // Define dummy Block-ID
  BlockID blockId;

  // Define and fill input with unique values
  DeviceMemory dInput(ctx, factory.bufferSize(BeamFormerTransposeKernel::INPUT_DATA));
  MultiDimArrayHostBuffer<fcomplex, 4> hInput(boost::extents[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL][NR_TABS][NR_POLARIZATIONS], ctx);

  for (size_t i = 0; i < hInput.num_elements(); ++i)
    hInput.origin()[i] = fcomplex(2 * i, 2 * i + 1);

  // Define output
  DeviceMemory dOutput(ctx, factory.bufferSize(BeamFormerTransposeKernel::OUTPUT_DATA));
  MultiDimArrayHostBuffer<fcomplex, 4> hOutput(boost::extents[NR_TABS][NR_POLARIZATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL], ctx);

  // Create kernel
  BeamFormerTransposeKernel::Buffers buffers(dInput, dOutput);
  std::auto_ptr<BeamFormerTransposeKernel> kernel(factory.create(stream, buffers));

  // Run kernel
  stream.writeBuffer(dInput, hInput, false);
  kernel->enqueue(blockId);
  stream.readBuffer(hOutput, dOutput, true);

  // Verify output
  for (size_t t = 0; t < NR_TABS; ++t)
    for (size_t p = 0; p < NR_POLARIZATIONS; ++p)
      for (size_t c = 0; c < NR_CHANNELS; ++c)
        for (size_t s = 0; s < NR_SAMPLES_PER_CHANNEL; ++s) {
          ASSERT(hOutput[t][p][c][s] == hInput[c][s][t][p]);
        }
}


int main()
{
  INIT_LOGGER("tTranspose2");
  try 
  {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } 
  catch (gpu::CUDAException& e) 
  {
    cerr << e.what() << endl;
    return 3;
  }

  // Create a stream
  Device device(0);
  Context ctx(device);
  Stream stream(ctx);

  // Run the test
  runTest(ctx, stream);

  return 0;
}


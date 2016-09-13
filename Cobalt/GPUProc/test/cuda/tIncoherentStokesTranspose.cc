//# tIncoherentStokesTranspose.cc: test Incoherent Stokes Transpose kernel
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
//# $Id: tIncoherentStokesTranspose.cc 27249 2013-11-05 17:39:42Z amesfoort $

#include <lofar_config.h>

#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/Kernels/IncoherentStokesTransposeKernel.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <CoInterface/BlockID.h>
#include <Common/Exception.h>

#include <boost/lexical_cast.hpp>
#include <complex>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace LOFAR::TYPES;

unsigned NR_CHANNELS = 61;
unsigned NR_SAMPLES_PER_CHANNEL = 47;
unsigned NR_STATIONS = 79;
unsigned NR_POLARIZATIONS = 2;

Exception::TerminateHandler t(Exception::terminate);

void runTest( gpu::Context &ctx, gpu::Stream &stream )
{
  // Use a dummy parset to construct default parameters.
  Parset ps;
  ps.updateSettings();

  IncoherentStokesTransposeKernel::Parameters params(ps);

  // Update the parameters ourselves.
  params.nrChannelsPerSubband = NR_CHANNELS;
  params.nrSamplesPerChannel = NR_SAMPLES_PER_CHANNEL;
  params.nrStations = NR_STATIONS;

  KernelFactory<IncoherentStokesTransposeKernel> factory(params);

  // Define dummy Block-ID
  BlockID blockId;

  // Define and fill input with unique values
  gpu::DeviceMemory
    dInput(ctx, 
           factory.bufferSize(IncoherentStokesTransposeKernel::INPUT_DATA));
  MultiDimArrayHostBuffer<fcomplex, 4>
    hInput(boost::extents
           [NR_STATIONS]
           [NR_CHANNELS]
           [NR_SAMPLES_PER_CHANNEL]
           [NR_POLARIZATIONS],
           ctx);
  for (size_t i = 0; i < hInput.num_elements(); ++i)
    hInput.data()[i] = fcomplex(2 * i, 2 * i + 1);

  // Define output and initialize to zero
  gpu::DeviceMemory 
    dOutput(ctx, 
            factory.bufferSize(IncoherentStokesTransposeKernel::OUTPUT_DATA));
  dOutput.set(0);
  MultiDimArrayHostBuffer<fcomplex, 4>
    hOutput(boost::extents
            [NR_STATIONS]
            [NR_POLARIZATIONS]
            [NR_SAMPLES_PER_CHANNEL]
            [NR_CHANNELS],
            ctx);
  fill(hOutput.data(), hOutput.data() + hOutput.num_elements(), fcomplex());

  // Create kernel
  IncoherentStokesTransposeKernel::Buffers buffers(dInput, dOutput);
  auto_ptr<IncoherentStokesTransposeKernel> 
    kernel(factory.create(stream, buffers));

  // Run kernel
  stream.writeBuffer(dInput, hInput, false);
  kernel->enqueue(blockId);
  stream.readBuffer(hOutput, dOutput, true);

  // Verify output if we're not profiling.
  if (!getenv("CUDA_PROFILING")) {
    for (size_t s = 0; s < NR_STATIONS; ++s)
      for (size_t p = 0; p < NR_POLARIZATIONS; ++p)
        for (size_t t = 0; t < NR_SAMPLES_PER_CHANNEL; ++t)
          for (size_t c = 0; c < NR_CHANNELS; ++c)
            ASSERTSTR(hOutput[s][p][t][c] == hInput[s][c][t][p],
                      "\n  hOutput[" << s << "][" << p << "][" << t << "][" << c <<
                      "] = " << hOutput[s][p][t][c] << 
                      ",    hInput[" << s << "][" << c << "][" << t << "][" << p <<
                      "] = " << hInput[s][c][t][p]);
    cout << "Test OK" << endl;
  } else {
    cout << "Output data not verified" << endl;
  }
}


int main()
{
  INIT_LOGGER("tIncoherentStokesTranspose");
  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    return 3;
  }

  // Create a stream
  gpu::Device device(0);
  gpu::Context ctx(device);
  gpu::Stream stream(ctx);

  // Run the test
  runTest(ctx, stream);

  return 0;
}


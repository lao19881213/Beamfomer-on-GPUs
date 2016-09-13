//# tBandPassCorrection.cc: test delay and bandpass CUDA kernel
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
//# $Id: tBandPassCorrection.cc 27477 2013-11-21 13:08:20Z loose $

#include <lofar_config.h>

#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>
#include <sstream>
#include <typeinfo>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <UnitTest++.h>

#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>

using namespace std;
using namespace LOFAR::Cobalt;

using LOFAR::i16complex;
using LOFAR::i8complex;

typedef complex<float> fcomplex;

gpu::Stream *stream;

// default compile definitions
const unsigned NR_STATIONS = 48;
const unsigned NR_CHANNELS_1 = 64;
const unsigned NR_CHANNELS_2 = 32;
const unsigned NR_SAMPLES_PER_CHANNEL = 32;
const unsigned NR_BITS_PER_SAMPLE = 8;
const unsigned NR_POLARIZATIONS = 2;

const unsigned nrResultVals =  NR_STATIONS * NR_CHANNELS_1 * NR_CHANNELS_2*
                                 NR_SAMPLES_PER_CHANNEL * NR_POLARIZATIONS;

// Initialize input AND output before calling runKernel().
// We copy both to the GPU, to make sure the final output is really from the
// kernel.  T is an LCS i*complex type, or complex<float> when #chnl > 1.
template <typename T>
void runKernel(gpu::Function kfunc,
               MultiDimArrayHostBuffer<fcomplex, 4> &outputData,
               MultiDimArrayHostBuffer<T,        5> &inputData,
               MultiDimArrayHostBuffer<float,    1> &bandPassFactors)
{
  gpu::Context ctx(stream->getContext());

  gpu::DeviceMemory devOutput         (ctx, outputData.size());
  gpu::DeviceMemory devInput          (ctx, inputData.size());
  gpu::DeviceMemory devBandPassFactors(ctx, bandPassFactors.size());

  kfunc.setArg(0, devOutput);
  kfunc.setArg(1, devInput);
  kfunc.setArg(2, devBandPassFactors);

  // 
  gpu::Grid globalWorkSize(NR_SAMPLES_PER_CHANNEL / 16,
                           NR_CHANNELS_2 / 16,
                           NR_STATIONS);

  gpu::Block localWorkSize(16, 16, 1);

  
  // Overwrite devOutput, so result verification is more reliable.
  stream->writeBuffer(devOutput,          outputData);
  stream->writeBuffer(devInput,           inputData);
  stream->writeBuffer(devBandPassFactors, bandPassFactors);
  stream->launchKernel(kfunc, globalWorkSize, localWorkSize);
  stream->synchronize(); // wait until transfer completes

  stream->readBuffer(outputData, devOutput);
  stream->synchronize(); // wait until transfer completes
}

gpu::Function initKernel(gpu::Context ctx, const CompileDefinitions& defs)
{
  // Compile to ptx. Copies the kernel to the current dir
  // (also the complex header, needed for compilation).
  string kernelPath("BandPassCorrection.cu");
  CompileFlags flags(defaultCompileFlags());
  vector<gpu::Device> devices(1, gpu::Device(0));
  string ptx(createPTX(kernelPath, defs, flags, devices));
  gpu::Module module(createModule(ctx, kernelPath, ptx));
  gpu::Function kfunc(module, "bandPassCorrection");

  return kfunc;
}

CompileDefinitions getDefaultCompileDefinitions()
{
  CompileDefinitions defs;

  defs["NR_STATIONS"] =
    boost::lexical_cast<string>(NR_STATIONS);
  defs["NR_CHANNELS_1"] =
    boost::lexical_cast<string>(NR_CHANNELS_1);
  defs["NR_CHANNELS_2"] =
    boost::lexical_cast<string>(NR_CHANNELS_2);
  defs["NR_SAMPLES_PER_CHANNEL"] =
    boost::lexical_cast<string>(NR_SAMPLES_PER_CHANNEL);
  defs["NR_POLARIZATIONS"] =
    boost::lexical_cast<string>(NR_POLARIZATIONS);
  defs["NR_BITS_PER_SAMPLE"] =
    boost::lexical_cast<string>(NR_BITS_PER_SAMPLE);
  defs["DO_BANDPASS_CORRECTION"] = "1";

  return defs;
}


// T is an LCS i*complex type, or complex<float> when #chnl > 1.
// It is the value type of the data input array.
template <typename T>
vector<fcomplex> runTest(const CompileDefinitions& compileDefs)
{
  gpu::Context ctx(stream->getContext());

  boost::scoped_ptr<MultiDimArrayHostBuffer<fcomplex, 4> > outputData;
  boost::scoped_ptr<MultiDimArrayHostBuffer<T,        5> > inputData;

  outputData.reset(
      new MultiDimArrayHostBuffer<fcomplex, 4>(boost::extents
                                               [NR_STATIONS]
                                               [NR_CHANNELS_1 * NR_CHANNELS_2]
                                               [NR_SAMPLES_PER_CHANNEL]
                                               [NR_POLARIZATIONS],
                                               ctx));

  inputData.reset(
      new MultiDimArrayHostBuffer<fcomplex, 5>(boost::extents
                                               [NR_STATIONS]
                                               [NR_POLARIZATIONS]
                                               [NR_CHANNELS_1]
                                               [NR_SAMPLES_PER_CHANNEL]
                                               [NR_CHANNELS_2],
                                               ctx));

  MultiDimArrayHostBuffer<float, 1> bandPassFactors(
      boost::extents [NR_CHANNELS_1 * NR_CHANNELS_2], ctx);

  // set inputs
  for (size_t i = 0; i < inputData->num_elements(); i++) {
    inputData->origin()[i].real() = 1.0f;
    inputData->origin()[i].imag() = 1.0f;
  }

  for (size_t i = 1; i < bandPassFactors.num_elements(); i++) {
    bandPassFactors.origin()[i] =  (i * 1.0) / bandPassFactors.num_elements();
  }

  // set output for proper verification later
  for (size_t i = 0; i < outputData->num_elements(); i++) {
    outputData->origin()[i].real() = 42.0f;
    outputData->origin()[i].imag() = 42.0f;
  }

  gpu::Function kfunc(initKernel(ctx, compileDefs));

  runKernel(kfunc, *outputData, *inputData, bandPassFactors);

  // Tests that use this function only check the first and last 2 output floats.
  vector<fcomplex> resultVals(2);
  resultVals[0] = outputData->origin()[0];
  resultVals[1] = outputData->origin()[nrResultVals-1];

  if(false)
  {
    for(size_t idx1 = 0; idx1 < NR_STATIONS; ++ idx1)
      for(size_t idx2 = 0; idx2 < NR_CHANNELS_1 * NR_CHANNELS_2; ++ idx2)
      {
        for(size_t idx3 = 0; idx3 < NR_SAMPLES_PER_CHANNEL; ++ idx3)
        {
          cout << "{" << (*outputData)[idx1][idx2][idx3][0]
               << ", " << (*outputData)[idx1][idx2][idx3][1] 
               << "}" ;
        }
        cout << endl;
      }
  }

  return resultVals;
}

TEST(BandPass)
{
  // ***********************************************************
  // Test if the bandpass correction factor is applied correctly in isolation
  CompileDefinitions defs(getDefaultCompileDefinitions());

  // The input samples are all ones. After correction, multiply with 2.
  // The first and the last complex values are retrieved. They should be scaled
  // with the bandPassFactor == 2
  vector<fcomplex> results(runTest<fcomplex>(defs)); // bandpass factor

  // The bandpassarray is filled with straigth slope from 0 untill 1
  // idx_in_array / array_elements
  CHECK_CLOSE(0.0, results[0].real(), 0.000001);

  // Last element is (array_elements - 1) / array_elements  index start at 0 until max-1
  CHECK_CLOSE((NR_CHANNELS_1 * NR_CHANNELS_2 * 1.0 - 1.0) / (NR_CHANNELS_1 * NR_CHANNELS_2), 
              results[1].real(), 0.000001);
}


gpu::Stream initDevice()
{
  // Set up device (GPU) environment
  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " GPU devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    exit(3); // test skipped
  }
  gpu::Device device(0);
  vector<gpu::Device> devices(1, device);
  gpu::Context ctx(device);
  gpu::Stream cuStream(ctx);

  return cuStream;
}

int main()
{
  INIT_LOGGER("tBandPassCorrection");

  // init global(s): device, context/stream.
  gpu::Stream strm(initDevice());
  stream = &strm;


  int exitStatus = UnitTest::RunAllTests();
  return exitStatus > 0 ? 1 : 0;
}


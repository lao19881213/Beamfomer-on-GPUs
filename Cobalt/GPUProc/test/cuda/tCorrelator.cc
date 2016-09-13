//# tCorrelator.cc: test correlator CUDA kernel
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
//# $Id: tCorrelator.cc 26790 2013-10-01 09:00:26Z mol $

#include <lofar_config.h>

#include <cstdlib>  // for rand()
#include <cmath>    // for abs()
#include <string>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <Common/Exception.h>
#include <Common/LofarLogger.h>

#include <CoInterface/MultiDimArray.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>

#include "../TestUtil.h"

using namespace std;
using namespace boost;
using namespace LOFAR::Cobalt::gpu;
using namespace LOFAR::Cobalt;
using LOFAR::Exception;

unsigned NR_STATIONS = 4;
unsigned NR_CHANNELS = 16;
unsigned NR_SAMPLES_PER_CHANNEL = 64;
unsigned NR_POLARIZATIONS = 2;
unsigned COMPLEX = 2;
unsigned NR_BASELINES = (NR_STATIONS * (NR_STATIONS + 1) / 2);

//
HostMemory runTest(gpu::Context ctx,
                   Stream cuStream,
                   float * inputData,
                   string function)
{
  string kernelFile = "Correlator.cu";

  cout << "\n==== runTest: function = " << function << " ====\n" << endl;

  // Get an instantiation of the default parameters
  CompileDefinitions definitions = CompileDefinitions();
  CompileFlags flags = defaultCompileFlags();

  // ****************************************
  // Compile to ptx
  // Set op string string pairs to be provided to the compiler as defines
  definitions["NVIDIA_CUDA"] = "";
  definitions["NR_STATIONS"] = lexical_cast<string>(NR_STATIONS);
  definitions["NR_CHANNELS"] = lexical_cast<string>(NR_CHANNELS);
  definitions["NR_SAMPLES_PER_CHANNEL"] = lexical_cast<string>(NR_SAMPLES_PER_CHANNEL);
  definitions["NR_POLARIZATIONS"] = lexical_cast<string>(NR_POLARIZATIONS);
  definitions["COMPLEX"] = lexical_cast<string>(COMPLEX);

  vector<Device> devices(1, ctx.getDevice());
  string ptx = createPTX(kernelFile, definitions, flags, devices);
  gpu::Module module(createModule(ctx, kernelFile, ptx));
  Function hKernel(module, function);   // c function this no argument overloading

  // *************************************************************
  // Create the data arrays
  size_t sizeCorrectedData = NR_STATIONS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARIZATIONS * COMPLEX * sizeof(float);
  DeviceMemory devCorrectedMemory(ctx, sizeCorrectedData);
  HostMemory rawCorrectedData = getInitializedArray(ctx, sizeCorrectedData, 0.0f);


  size_t sizeVisibilitiesData = NR_BASELINES * NR_CHANNELS * NR_POLARIZATIONS * NR_POLARIZATIONS * COMPLEX * sizeof(float);
  DeviceMemory devVisibilitiesMemory(ctx, sizeVisibilitiesData);
  HostMemory rawVisibilitiesData = getInitializedArray(ctx, sizeVisibilitiesData, 42.0f);
  cuStream.writeBuffer(devVisibilitiesMemory, rawVisibilitiesData);

  //copy the input received as argument to the input array
  float *rawDataPtr = rawCorrectedData.get<float>();
  for (unsigned idx = 0; idx < sizeCorrectedData / sizeof(float); ++idx)
    rawDataPtr[idx] = inputData[idx];
  cuStream.writeBuffer(devCorrectedMemory, rawCorrectedData);

  // ****************************************************************************
  // Run the kernel on the created data
  hKernel.setArg(0, devVisibilitiesMemory);
  hKernel.setArg(1, devCorrectedMemory);

  // Calculate the number of threads in total and per blovk
  unsigned nrBlocks = NR_BASELINES;
  unsigned nrPasses = (nrBlocks + 1024 - 1) / 1024;
  unsigned nrThreads = (nrBlocks + nrPasses - 1) / nrPasses;
  unsigned nrUsableChannels = 15;
  Grid globalWorkSize(nrPasses, nrUsableChannels, 1);
  Block localWorkSize(nrThreads, 1,1);

  // Run the kernel
  cuStream.synchronize(); // assure memory is copied
  cuStream.launchKernel(hKernel, globalWorkSize, localWorkSize);
  cuStream.synchronize(); // assure that the kernel is finished

  // Copy output vector from GPU buffer to host memory.
  cuStream.readBuffer(rawVisibilitiesData, devVisibilitiesMemory);
  cuStream.synchronize(); //assure copy from device is done

  return rawVisibilitiesData;
}

Exception::TerminateHandler t(Exception::terminate);

int main()
{
  INIT_LOGGER("tCorrelator");

  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    return 3;
  }

  // Create a default context
  gpu::Device device(0);
  gpu::Context ctx(device);
  Stream cuStream(ctx);

  // Create data members
  MultiDimArray<float, 3> inputData(boost::extents[NR_STATIONS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL * NR_POLARIZATIONS * COMPLEX]);
  MultiDimArray<float, 3> outputData(boost::extents[NR_BASELINES][NR_CHANNELS][NR_POLARIZATIONS * NR_POLARIZATIONS * COMPLEX]);
  float * outputOnHostPtr;

  const char * kernel_functions[] = {
    "correlate", "correlate_2x2", "correlate_3x3", "correlate_4x4"
  };
  unsigned nr_kernel_functions =
    sizeof(kernel_functions) / sizeof(kernel_functions[0]);

  for (unsigned func_idx = 0; func_idx < nr_kernel_functions; func_idx++) 
  {
    cerr << kernel_functions[func_idx] << endl;
    const char* function = kernel_functions[func_idx];

    // ***********************************************************
    // Baseline test: If all input data is zero the output should be zero
    // The output array is initialized with 42s
    for (unsigned idx = 0; idx < inputData.num_elements(); ++idx)
      inputData.origin()[idx] = 0;

    HostMemory outputOnHost = runTest(ctx, cuStream, inputData.origin(), function);

    // Copy the output data to a local array
    outputOnHostPtr = outputOnHost.get<float>();
    for (unsigned idx = 0; idx < outputData.num_elements(); ++idx)
      outputData.origin()[idx] = outputOnHostPtr[idx];

    // Now validate the outputdata
    for (unsigned idx_baseline = 0; idx_baseline < NR_BASELINES; ++idx_baseline)
    {
      cerr << "baseline: " << idx_baseline << endl;
      for (unsigned idx_channels = 0; idx_channels < NR_CHANNELS; ++idx_channels)
      {
        cerr << idx_channels << " : ";
        for (unsigned idx = 0; idx < NR_POLARIZATIONS * NR_POLARIZATIONS * COMPLEX; ++idx)
        {
          float sample = outputData[idx_baseline][idx_channels][idx];
          cerr << sample << ", ";
          if (idx_channels != 0)
            if (sample != 0)
            {
              cerr << "Non-zero number encountered while all input data were zero." << endl;
              return 1;
            }
        }
        cerr << endl;
      }
    }

    // ***********************************************************
    // To test if the kernel is working we try to correlate a number of delayed
    // random initialize channels.
    // With zero delay the output should be the highest. A fringe :D

    // 1. First create a random channel with a length that is large enough
    // It should be length NR_SAMPLES_PER_CHANNEL plus padding at both side to encompass the delay
    unsigned padding = 7; // We have 15 channels with content 2 * 7 delays + delay 0
    unsigned lengthRandomData = NR_SAMPLES_PER_CHANNEL * NR_POLARIZATIONS * COMPLEX + 2 * padding * 4;
    vector<float> randomInputData(lengthRandomData);

    // Create the random signal, seed random generator with zero
    srand (0);
    for (unsigned idx = 0; idx < randomInputData.size(); ++idx)
      randomInputData[idx] = ((rand() % 1024 + 1) * 1.0  // make float
                              - 512.0) / 512.0;          // centre around zero and normalize

    // Fill the input array channels
    // channel 8 is the non delayed channel and should correlate
    for (unsigned idx_station = 0; idx_station < 2; ++idx_station)
    {
      cerr << "idx_station: " << idx_station << endl;
      for (unsigned idx_channel = 0; idx_channel < 16; ++idx_channel)
      {
        for (unsigned idx_datapoint = 0;
             idx_datapoint< NR_SAMPLES_PER_CHANNEL * NR_POLARIZATIONS * COMPLEX;
             ++idx_datapoint)
        {
          // Pick from the random array the same number of samples
          // But with an offset depending on the channel number
          unsigned padding_offset = 32;
          int padding = padding_offset - idx_channel * 4;       // station 1 is filled with delayed signal

          // Station zero is anchored as the middle and is filled with the baseline signal
          if (idx_station == 0)
            padding = padding_offset - 8 * 4;

          // Get the index in the random signal array
          unsigned idx_randomdata = padding_offset + padding + idx_datapoint;

          //assign the signal;
          inputData[idx_station][idx_channel][idx_datapoint] = randomInputData[idx_randomdata];
          if (idx_datapoint < 16)        // plot first part if the input signal for debugging purpose
            cerr << inputData[idx_station][idx_channel][idx_datapoint] << " : ";
        }
        cerr << endl;
      }
    }

    // Run the kernel
    outputOnHost = runTest(ctx, cuStream, inputData.origin(), function);

    // Copy the output data to a local array
    outputOnHostPtr = outputOnHost.get<float>();
    for (unsigned idx = 0; idx < outputData.num_elements(); ++idx)
      outputData.origin()[idx] = outputOnHostPtr[idx];

    // Target value for correlation channel
    float targetValues[8] = {36.2332, 0, -7.83033, 3.32368, -7.83033, -3.32368, 42.246, 0};

    // print the contents of the output array for debugging purpose
    for (unsigned idx_baseline = 0; idx_baseline < NR_BASELINES; ++idx_baseline)
    {
      cerr << "baseline: " << idx_baseline << endl;
      for (unsigned idx_channels = 0; idx_channels < NR_CHANNELS; ++idx_channels)
      {
        cerr << idx_channels << " : ";
        for (unsigned idx = 0; idx < NR_POLARIZATIONS * NR_POLARIZATIONS * COMPLEX; ++idx)
        {
          float sample = outputData[idx_baseline][idx_channels][idx];
          if (idx_baseline == 1 && idx_channels == 8)
          {

            //validate that the correct value is found
            if (abs(sample - targetValues[idx]) > 0.0001)
            {
              cerr << "The correlated data found was not within an acceptable delta:" << endl
                   << "Expected: " << sample << endl
                   << "Found: " << targetValues[idx] << endl
                   << "Difference: " << sample - targetValues[idx]
                   << "  Delta: " << 0.0001 << endl;

              return 1;
           }
          }
          cerr << sample << ", ";
        }
        cerr << endl;
      }
    }
    
    // ***********************************************************
    // test 3: If all input data is zero the output should be zero
    // except a specific set of values
    cerr << "Length input data:" << inputData.num_elements() << endl;
    for (unsigned idx = 0; idx < inputData.num_elements(); ++idx)
      inputData.origin()[idx] = 0;

    // insert some values at specific locations in the input matrix
    unsigned timestep = NR_POLARIZATIONS*COMPLEX;
    // [0][5][7][0][0] = 2
    // [0][5][7][0][1] = 3
    inputData[0][5][timestep * 7] = 2;
    inputData[0][5][timestep * 7 + 1] = 3;
    // [1][5][7][1][0] = 4
    // [1][5][7][1][1] = 5
    inputData[1][5][timestep * 7 + NR_POLARIZATIONS] = 4;
    inputData[1][5][timestep * 7 + NR_POLARIZATIONS + 1] = 5;

    outputOnHost = runTest(ctx, cuStream, inputData.origin(), function);

    // Copy the output data to a local array
    outputOnHostPtr = outputOnHost.get<float>();
    for (unsigned idx = 0; idx < outputData.num_elements(); ++idx)
      outputData.origin()[idx] = outputOnHostPtr[idx];

    // Now validate the outputdata
    for (unsigned idx_baseline = 0; idx_baseline < NR_BASELINES; ++idx_baseline)
    {
      cerr << "baseline: " << idx_baseline << endl;

      // skip channel 0
      for (unsigned idx_channels = 1; idx_channels < NR_CHANNELS; ++idx_channels)
      {
        cerr << idx_channels << " : ";
        for (unsigned idx = 0; idx < NR_POLARIZATIONS * NR_POLARIZATIONS * COMPLEX; ++idx)
        {
          float sample = outputData[idx_baseline][idx_channels][idx];
          float expected = 0.0f;

          cerr << idx << ":" << sample << ", ";

          // We need to find 4 specific indexes with values:
          // THe output location of the values does not change with differing input size
          if (idx_baseline == 0 &&  idx_channels == 5 && idx == 0)
            expected = 13.0f;
          
          if (idx_baseline == 1 &&  idx_channels == 5 && idx == 2)
            expected = 23.0f;

          if (idx_baseline == 1 &&  idx_channels == 5 && idx == 3)
            expected = 2.0f;
          
          if (idx_baseline == 2 &&  idx_channels == 5 && idx == 6)
            expected = 41.0f;

          if (sample != expected)
          {
            cerr << "Unexpected number encountered: got " << sample << " but expected " << expected << endl;
            return 1;
          }
        }
        cerr << endl;
      }
    }


  } // for func_idx

  return 0;
}

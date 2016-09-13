//# tBeamFormer.cc: test BeamFormer CUDA kernel
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
//# $Id: tBeamFormer.cc 26671 2013-09-25 00:51:52Z amesfoort $

#include <lofar_config.h>

#include <cstdlib>
#include <cmath>
#include <complex>
#include <string>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <Common/Exception.h>
#include <Common/LofarLogger.h>

#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>

#include "../fpequals.h"
#include "../TestUtil.h"

using namespace std;
using namespace boost;
using namespace LOFAR::Cobalt::gpu;
using namespace LOFAR::Cobalt;

// The tests succeed for different values of stations, channels, samples and TABs.
unsigned NR_STATIONS = 4;  
unsigned NR_CHANNELS = 4;
unsigned NR_SAMPLES_PER_CHANNEL = 4;
unsigned NR_SAPS = 2;
unsigned NR_TABS = 4;
unsigned NR_POLARIZATIONS = 2;


// Length in elements of the data arrays
size_t lengthDelaysData = NR_SAPS * NR_STATIONS * NR_TABS; // double
size_t lengthBandPassCorrectedData = NR_STATIONS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARIZATIONS; // complex<float>
size_t lengthComplexVoltagesData = NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_TABS * NR_POLARIZATIONS; // complex<float>


LOFAR::Exception::TerminateHandler t(LOFAR::Exception::terminate);


void exit_with_print(const complex<float> *outputOnHostPtr)
{
  // Plot the output of the kernel in a readable manner
  unsigned size_channel = NR_SAMPLES_PER_CHANNEL * NR_TABS * NR_POLARIZATIONS;
  unsigned size_sample = NR_TABS * NR_POLARIZATIONS;
  unsigned size_tab = NR_POLARIZATIONS;
  for (unsigned idx_channel = 0; idx_channel < NR_CHANNELS; ++ idx_channel)
  {
    cout << "idx_channel: " << idx_channel << endl;
    for (unsigned idx_samples = 0; idx_samples < NR_SAMPLES_PER_CHANNEL; ++ idx_samples)
    {
      cout << "idx_samples " << idx_samples << ": ";
      for (unsigned idx_tab = 0; idx_tab < NR_TABS; ++ idx_tab)
      {
        unsigned index_data_base = size_channel * idx_channel + 
                                   size_sample * idx_samples +
                                   size_tab * idx_tab ;
        cout << outputOnHostPtr[index_data_base] <<          // X pol
                outputOnHostPtr[index_data_base + 1] << " "; // Y pol
      }
      cout << endl;
    }
    cout << endl;
  }
  
  std::exit(1);
}


HostMemory runTest(Context ctx,
                   Stream cuStream,
                   double * delaysData,
                   complex<float> * bandPassCorrectedData,
                   complex<float> * complexVoltagesData,
                   double subbandFrequency,
                   unsigned sap,
                   string function,
                   float weightCorrection,
                   double subbandBandwidth)
{
  string kernelFile = "BeamFormer.cu";

  cout << "\n==== runTest: function = " << function << " ====\n" << endl;

  // Get an instantiation of the default parameters
  CompileFlags flags = CompileFlags();
  CompileDefinitions definitions = CompileDefinitions();
  
  // ****************************************
  // Compile to ptx
  // Set op string string pairs to be provided to the compiler as defines
  definitions["NVIDIA_CUDA"] = "";
  definitions["NR_STATIONS"] = lexical_cast<string>(NR_STATIONS);
  definitions["NR_CHANNELS"] = lexical_cast<string>(NR_CHANNELS);
  definitions["NR_SAMPLES_PER_CHANNEL"] = lexical_cast<string>(NR_SAMPLES_PER_CHANNEL);
  definitions["NR_SAPS"] = lexical_cast<string>(NR_SAPS);
  definitions["NR_TABS"] = lexical_cast<string>(NR_TABS);
  definitions["NR_POLARIZATIONS"] = lexical_cast<string>(NR_POLARIZATIONS);
  definitions["WEIGHT_CORRECTION"] = str(boost::format("%.7ff") % weightCorrection);
  definitions["SUBBAND_BANDWIDTH"] = str(boost::format("%.7f") % subbandBandwidth);

  vector<Device> devices(1, ctx.getDevice());
  string ptx = createPTX(kernelFile, definitions, flags, devices);
  Module module(createModule(ctx, kernelFile, ptx));
  Function hKernel(module, function);   // c function this no argument overloading

  // *************************************************************
  // Create the data arrays
  size_t sizeDelaysData = lengthDelaysData * sizeof(double);
  DeviceMemory devDelaysMemory(ctx, sizeDelaysData);
  HostMemory rawDelaysData = getInitializedArray(ctx, sizeDelaysData, 1.0f);
  double *rawDelaysPtr = rawDelaysData.get<double>();
  for (unsigned idx = 0; idx < lengthDelaysData; ++idx)
    rawDelaysPtr[idx] = delaysData[idx];
  cuStream.writeBuffer(devDelaysMemory, rawDelaysData);

  size_t sizeBandPassCorrectedData = lengthBandPassCorrectedData * sizeof(complex<float>);
  DeviceMemory devBandPassCorrectedMemory(ctx, sizeBandPassCorrectedData);
  HostMemory rawBandPassCorrectedData = getInitializedArray(ctx, sizeBandPassCorrectedData, 2.0f);
  complex<float> *rawBandPassCorrectedPtr = rawBandPassCorrectedData.get<complex<float> >();
  for (unsigned idx = 0; idx < lengthBandPassCorrectedData; ++idx)
    rawBandPassCorrectedPtr[idx] = bandPassCorrectedData[idx];
  cuStream.writeBuffer(devBandPassCorrectedMemory, rawBandPassCorrectedData);

  size_t sizeComplexVoltagesData = lengthComplexVoltagesData * sizeof(complex<float>);
  DeviceMemory devComplexVoltagesMemory(ctx, sizeComplexVoltagesData);
  HostMemory rawComplexVoltagesData = getInitializedArray(ctx, sizeComplexVoltagesData, 3.0f);
  complex<float> *rawComplexVoltagesPtr = rawComplexVoltagesData.get<complex<float> >();
  for (unsigned idx = 0; idx < lengthComplexVoltagesData; ++idx)
    rawComplexVoltagesPtr[idx] = complexVoltagesData[idx];
  // Write output content.
  cuStream.writeBuffer(devComplexVoltagesMemory, rawComplexVoltagesData);

  // ****************************************************************************
  // Run the kernel on the created data
  hKernel.setArg(0, devComplexVoltagesMemory);
  hKernel.setArg(1, devBandPassCorrectedMemory);
  hKernel.setArg(2, devDelaysMemory);
  hKernel.setArg(3, subbandFrequency);
  hKernel.setArg(4, sap);

  // Calculate the number of threads in total and per block
  Grid globalWorkSize(1, 1, 1);
  Block localWorkSize(NR_POLARIZATIONS, NR_TABS, NR_CHANNELS);

  // Run the kernel
  cuStream.launchKernel(hKernel, globalWorkSize, localWorkSize);

  // Copy output vector from GPU buffer to host memory.
  cuStream.readBuffer(rawComplexVoltagesData, devComplexVoltagesMemory, true);

  return rawComplexVoltagesData; 
}

int main()
{
  INIT_LOGGER("tBeamFormer");
  const char function[] = "beamFormer";

  try 
  {
    Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } 
  catch (CUDAException& e) 
  {
    cerr << e.what() << endl;
    return 3;
  }

  // Create a default context
  Device device(0);
  Context ctx(device);
  Stream cuStream(ctx);

  // Define the input and output arrays
  // ************************************************************* 
  complex<float>* complexVoltagesData = new complex<float>[lengthComplexVoltagesData];
  complex<float>* bandPassCorrectedData = new complex<float>[lengthBandPassCorrectedData];
  double * delaysData= new double[lengthDelaysData];
  complex<float>* outputOnHostPtr;

  complex<float> refVal;

  double subbandFrequency = 1.5e8; // Hz
  unsigned sap = 0;

  float weightCorrection = 1.0f;
  double subbandBandwidth = 200e3; // Hz

  const float POISONF = 42.0f; // to init output data for easy recognition if not written to by the kernel

  // ***********************************************************
  // Baseline test 1: If all delays data is zero and input (1, 1) (and weightCorrection=1),
  // then the output must be all (NR_STATIONS, NR_STATIONS). (See below and/or kernel why.)
  // The output array is initialized with 42s
  cout << "test 1" << endl;
  for (unsigned idx = 0; idx < lengthComplexVoltagesData; ++idx)
  {
    complexVoltagesData[idx].real(POISONF);
    complexVoltagesData[idx].imag(POISONF);
  }

  for (unsigned idx = 0; idx < lengthBandPassCorrectedData; ++idx)
  {
    bandPassCorrectedData[idx].real(1.0f);
    bandPassCorrectedData[idx].imag(1.0f);
  }

  for (unsigned idx = 0; idx < lengthDelaysData; ++idx)
  {
    delaysData[idx] = 0.0;
  }

  HostMemory outputOnHost1 = runTest(ctx, cuStream, delaysData,
                                     bandPassCorrectedData,
                                     complexVoltagesData,
                                     subbandFrequency, sap,
                                     function,
                                     weightCorrection, subbandBandwidth);

  /*
   * Reference output calculation:
   * The kernel calculates the beamformer weights per station from the delays (and the weightCorrection).
   * Since delays is 0, phi will be -M_PI * delay * channel_frequency = 0.
   * The complex weight is then (cos(phi), sin(phi)) * weightCorrection = (1, 0) * 1.
   * The samples (all (1, 1)) are then complex multiplied with the weights and summed for all (participating) stations.
   * The complex mul gives (1, 1). Summing NR_STATIONS of samples gives (NR_STATIONS, NR_STATIONS) for all samples.
   */
  refVal.real((float)NR_STATIONS);
  refVal.imag((float)NR_STATIONS);

  // Validate the returned data array
  outputOnHostPtr = outputOnHost1.get<complex<float> >();
  for (size_t idx = 0; idx < lengthComplexVoltagesData; ++idx)
  {
    if (!fpEquals(outputOnHostPtr[idx], refVal))
    {
      cerr << "The data returned by the kernel should be all (NR_STATIONS, NR_STATIONS): All input is (1, 1) and all delays are zero.";
      exit_with_print(outputOnHostPtr);
    }
  }

  // ***********************************************************
  // Baseline test 2: If all input data is zero the output should be zero while the delays are non zero
  // The output array is initialized with 42s
  cout << "test 2" << endl;
  for (unsigned idx = 0; idx < lengthComplexVoltagesData; ++idx)
  {
    complexVoltagesData[idx].real(POISONF);
    complexVoltagesData[idx].imag(POISONF);
  }

  for (unsigned idx = 0; idx < lengthBandPassCorrectedData; ++idx)
  {
    bandPassCorrectedData[idx].real(0.0f);
    bandPassCorrectedData[idx].imag(0.0f);
  }

  for (unsigned idx = 0; idx < lengthDelaysData; ++idx)
  {
    delaysData[idx] = 1.0;
  }

  HostMemory outputOnHost2 = runTest(ctx, cuStream, delaysData,
                                     bandPassCorrectedData,
                                     complexVoltagesData,
                                     subbandFrequency, sap,
                                     function,
                                     weightCorrection, subbandBandwidth);

  refVal.real(0.0f);
  refVal.imag(0.0f);

  // Validate the returned data array
  outputOnHostPtr = outputOnHost2.get<complex<float> >();
  for (size_t idx = 0; idx < lengthComplexVoltagesData; ++idx)
    if (!fpEquals(outputOnHostPtr[idx], refVal))
    {
      cerr << "The data returned by the kernel should be all zero: All inputs are zero";
      exit_with_print(outputOnHostPtr);
    }


  // ***********************************************************
  // Test 3: Test all, but cause reasonably "simple" reference output.
  // Set an unrealistic sb freq and bw, such that the ref output becomes a simple, non-zero nr and the same for all channels.
  // Make sure the mult of (any channelFrequency, delay) ends up at an -(integer+(1/12)). The phi will then be a 2N+pi/6,
  // whose cos and sin are then "simple", non-zero. The phase shift to (complex) mult the samples with is ( cos(phi), sin(phi) ),
  // possibly times some weight correction. Then sum over all stations, i.e. all equal, so * NR_STATIONS.

  cout << "test 3" << endl;
  for (unsigned idx = 0; idx < lengthComplexVoltagesData; ++idx)
  {
    complexVoltagesData[idx].real(POISONF);
    complexVoltagesData[idx].imag(POISONF);
  }

  const complex<float> inputVal((float)sqrt(3.0), 2.0f); // to square a sqrt(3) later
  for (unsigned idx = 0; idx < lengthBandPassCorrectedData; ++idx)
  {
    bandPassCorrectedData[idx].real(inputVal.real());
    bandPassCorrectedData[idx].imag(inputVal.imag());
  }

  for (unsigned idx = 0; idx < lengthDelaysData; ++idx)
  {
    delaysData[idx] = -0.5; // to get rid of the -2 in the *-2*pi, to get a nice phi
  }

  subbandFrequency = 200 + 1.0/6.0; // to get a nice phi of N+pi/6 with N some 
  subbandBandwidth = 4.0 * NR_CHANNELS; // on NR_CHANNELS=4 gives chnl freqs {192+1/6, 196+1/6, 200+1/6, 204+1/6}
  weightCorrection = 2.0f;

  complex<double> cosSinPhi(
                    cos(M_PI / 6.0),  // 0.5*sqrt(3.0) (~ 0.86602540378443864676)
                    sin(M_PI / 6.0)); // 0.5

  // refVal.real() is a nice 3*NR_STATIONS; imag() ends up at the just acceptable ref output nr of 3*sqrt(3)*NR_STATIONS
  refVal = cosSinPhi * (double)weightCorrection * complex<double>(inputVal.real(), inputVal.imag()) * (double)NR_STATIONS;
  cout << "Expected ref output for test 3: " << refVal << endl;

  HostMemory outputOnHost3 = runTest(ctx, cuStream, delaysData,
                                     bandPassCorrectedData,
                                     complexVoltagesData,
                                     subbandFrequency, sap,
                                     function,
                                     weightCorrection, subbandBandwidth);

  // Validate the returned data array
  outputOnHostPtr = outputOnHost3.get<complex<float> >();
  for (size_t idx = 0; idx < lengthComplexVoltagesData; ++idx)
    if (!fpEquals(outputOnHostPtr[idx], refVal))
    {
      cerr << "all the data returned by the kernel should be " << refVal << endl;
      exit_with_print(outputOnHostPtr);
    }
    

  delete [] complexVoltagesData;
  delete [] bandPassCorrectedData;
  delete [] delaysData;

  return 0;
}


//# tCoherentStokes.cc: test BeamFormer CUDA kernel
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
//# $Id: tCoherentStokes.cc 25747 2013-07-24 13:12:39Z klijn $

#include <lofar_config.h>

#include <string>
#include <iostream>

#include <boost/lexical_cast.hpp>

#include <Common/Exception.h>
#include <Common/LofarLogger.h>

#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <UnitTest++.h>

#include "TestUtil.h"

using namespace std;
using namespace boost;
using namespace LOFAR::Cobalt::gpu;
using namespace LOFAR::Cobalt;
using LOFAR::Exception;

// Wrapper for the coherent stokes kernel
// Performs the runtime compilation and management.
struct CoherentStokesTester
{
  const char* function;
  // Create a default context
  Device device;
  Context ctx;
  Stream cuStream;


  // Data members
  unsigned NR_POLARIZATIONS;
  unsigned COMPLEX;
  unsigned NR_CHANNELS;
  unsigned NR_SAMPLES_PER_CHANNEL;
  unsigned TIME_PARALLEL_FACTOR;  // cannot be more then NR_SAMPLES_PER_CHANNEL / INTEGRATION_SIZE
  unsigned NR_TABS; 
  unsigned NR_COHERENT_STOKES;
  unsigned INTEGRATION_SIZE;

public:
  //sizes need to be public to allow usage by external parties
  size_t lengthInputData;
  size_t lengthOutputData;

  CoherentStokesTester(
    unsigned iNR_CHANNELS = 1,
    unsigned iNR_SAMPLES_PER_CHANNEL = 2,
    unsigned iTIME_PARALLEL_FACTOR =2, 
    unsigned iNR_TABS = 1,
    unsigned iNR_COHERENT_STOKES = 1,
    unsigned iINTEGRATION_SIZE = 1)
    :
      function("coherentStokes"),
      device(0),
      ctx(device),
      cuStream(ctx),
      NR_POLARIZATIONS(2),
      COMPLEX(2),
      NR_CHANNELS(iNR_CHANNELS),
      NR_SAMPLES_PER_CHANNEL(iNR_SAMPLES_PER_CHANNEL),
      TIME_PARALLEL_FACTOR(iTIME_PARALLEL_FACTOR), 
      NR_TABS(iNR_TABS),
      NR_COHERENT_STOKES(iNR_COHERENT_STOKES),
      INTEGRATION_SIZE(iINTEGRATION_SIZE)
    {
      
      // Create the data arrays
      lengthInputData = NR_TABS * NR_POLARIZATIONS * NR_SAMPLES_PER_CHANNEL * NR_CHANNELS * COMPLEX;     
      lengthOutputData = NR_TABS * NR_COHERENT_STOKES *  (NR_SAMPLES_PER_CHANNEL / INTEGRATION_SIZE) * NR_CHANNELS;

      //multiply with two to allow for out of bound checking
      lengthOutputData *= 2;

    }

  HostMemory runTest(
    float * inputData,
    float * outputData)
  {
    // THe constanst might have been changed, recalculate the length
    lengthInputData = NR_TABS * NR_POLARIZATIONS * NR_SAMPLES_PER_CHANNEL * NR_CHANNELS * COMPLEX;
    lengthOutputData = NR_TABS * NR_COHERENT_STOKES *  (NR_SAMPLES_PER_CHANNEL / INTEGRATION_SIZE) * NR_CHANNELS;

    //multiply with two to allow for out of bound checking
    lengthOutputData *= 2;

    string kernelFile = "CoherentStokes.cu";

    cout << "\n==== runTest: function = " << function << " ====\n" << endl;

    // Get an instantiation of the default parameters
    CompileFlags flags = CompileFlags();
    CompileDefinitions definitions = CompileDefinitions();

    // ****************************************
    // Compile to ptx
    // Set op string string pairs to be provided to the compiler as defines
    definitions["NVIDIA_CUDA"] = "";
    definitions["NR_CHANNELS"] = lexical_cast<string>(NR_CHANNELS);
    definitions["NR_SAMPLES_PER_CHANNEL"] = lexical_cast<string>(NR_SAMPLES_PER_CHANNEL);
    definitions["NR_TABS"] = lexical_cast<string>(NR_TABS);
    definitions["NR_POLARIZATIONS"] = lexical_cast<string>(NR_POLARIZATIONS);
    definitions["COMPLEX"] = lexical_cast<string>(COMPLEX);
    definitions["NR_COHERENT_STOKES"] = lexical_cast<string>(NR_COHERENT_STOKES);
    definitions["INTEGRATION_SIZE"] = lexical_cast<string>(INTEGRATION_SIZE);
    definitions["TIME_PARALLEL_FACTOR"] = lexical_cast<string>(TIME_PARALLEL_FACTOR);

    vector<Device> devices(1, ctx.getDevice());
    string ptx = createPTX(kernelFile, definitions, flags, devices);
    Module module(createModule(ctx, kernelFile, ptx));
    Function hKernel(module, function);   

    // *************************************************************
    // Create the data arrays

    size_t sizeInputData= lengthInputData * sizeof(float);
    DeviceMemory devInputMemory(ctx, sizeInputData);
    HostMemory rawInputData = getInitializedArray(ctx, sizeInputData, 2.0f);
    float *rawInputPtr = rawInputData.get<float>();
    for (unsigned idx = 0; idx < lengthInputData; ++idx)
      rawInputPtr[idx] = inputData[idx];
    cuStream.writeBuffer(devInputMemory, rawInputData);

    // Remember to increase the length of the output data to check for
    // out of bounds writes!!
    size_t sizeOutputData = lengthOutputData * sizeof(float);
    DeviceMemory devOutputMemory(ctx, sizeOutputData);
    HostMemory rawOutputData = getInitializedArray(ctx, sizeOutputData, 3.0f);
    float *rawOutputPtr = rawOutputData.get<float>();
    for (unsigned idx = 0; idx < lengthOutputData; ++idx)
      rawOutputPtr[idx] = outputData[idx];
    // Write output content.
    cuStream.writeBuffer(devOutputMemory, rawOutputData);

    // ****************************************************************************
    // Run the kernel on the created data
    hKernel.setArg(0, devOutputMemory);
    hKernel.setArg(1, devInputMemory);

    // Calculate the number of threads in total and per block
    Grid globalWorkSize(NR_CHANNELS, TIME_PARALLEL_FACTOR, NR_TABS);
    Block localWorkSize(NR_CHANNELS, TIME_PARALLEL_FACTOR, NR_TABS);

    // Run the kernel
    cuStream.synchronize(); // assure memory is copied
    cuStream.launchKernel(hKernel, globalWorkSize, localWorkSize);
    cuStream.synchronize(); // assure that the kernel is finished

    // Copy output vector from GPU buffer to host memory.
    cuStream.readBuffer(rawOutputData, devOutputMemory);
    cuStream.synchronize(); //assure copy from device is done

    return rawOutputData; 
  }
};

// Print the contents send and received from the gpu
void exit_with_print(float *inputData,
                     float *outputData,
                     size_t input_size,                        
                     size_t output_size)
  {
    // Plot the output of the kernel in a readable manner
    cout << "input send to GPU:" << endl;
    cout << "IDX 0: " << inputData[0];
    for ( size_t idx = 1; idx < input_size ; ++idx)
    {
      if (( idx % 20) == 0)
        cout << endl << "IDX " << idx << ": " ;
      cout << "," << inputData[idx];
    }
    cout << endl << endl << "output received from gpu:" << endl;
    cout << "IDX 0: " << outputData[0];
    for ( size_t idx = 1; idx < output_size ; ++idx)
    {
      if (( idx % 20) == 0)
        cout << endl << "IDX " << idx << ": " ;
      cout << "," << outputData[idx];
    }
    cout << endl;
  }

TEST(BasicZeroTest)
{
  // ***********************************************************
  // Baseline test: If all weight data is zero the output should be zero
  // outof bound mem should be untouched
  // The output array is initialized with 42s
  // With all constant on the smallest == default size
  // 2 samples
  // integrated over 2 samples
  // 1 coherent stokes

  // Create the testharnass with the system under test
  CoherentStokesTester sut;
  float salt = 42.0f;
  // Get the input and output arrays
  // ************************************************************* 
  float * inputData = new float[sut.lengthInputData];
  float * outputData = new float[sut.lengthOutputData]; 
  float * outputOnHostPtr;

  //initialize
  for (unsigned idx = 0; idx < sut.lengthInputData; ++idx)
    inputData[idx] = 0.0f;

  for (unsigned idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = salt;

  // excersize the system under test
  HostMemory outputOnHost = sut.runTest(inputData, outputData);
 
  //move data in local array
  outputOnHostPtr = outputOnHost.get<float>();
  for (size_t idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = outputOnHostPtr[idx] ;

  // Validate the returned data array
  for (size_t idx = 0; idx < sut.lengthOutputData / 2; ++idx)
    if (outputData[idx]  != 0)
    {
      cout << "encountered incorrect output on idx: " << idx << endl;
      cout << "expected: " << 0 << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData / 2);  //do not print salt
      //fail the current test

      CHECK(false);
      return;
    }
  
  // validate the output bound memory: it should al be the salt value
  for (size_t idx = sut.lengthOutputData / 2; idx < sut.lengthOutputData; ++idx)
    if (outputData[idx]  != salt )
    {
      cout << "encountered memory writes on invalid memory idx: " << idx << endl;
      cout << "expected: " << salt << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData);
      //fail the current test

      CHECK(false);
      return;
    }

  delete [] inputData;
  delete [] outputData;
}


TEST(LargeBasicZeroTest)
{
  // ***********************************************************
  // Baseline test: If all weight data is zero the output should be zero
  // outof bound mem should be untoughed
  // The output array is initialized with 42s
  // Create the testharnass with the system under test
  CoherentStokesTester sut(
    16,  //NR_CHANNELS
    2048,  //NR_SAMPLES_PER_CHANNEL
    4,  //TIME_PARALLEL_FACTOR
    16,  //NR_TABS
    4,  //NR_COHERENT_STOKES 
    64); //INTEGRATION_SIZE
  float salt = 42.0f;
  // Get the input and output arrays
  // ************************************************************* 
  float * inputData = new float[sut.lengthInputData];
  float * outputData = new float[sut.lengthOutputData]; 
  float * outputOnHostPtr;

  //initialize
  for (unsigned idx = 0; idx < sut.lengthInputData; ++idx)
    inputData[idx] = 0.0f;

  for (unsigned idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = salt;

  // excersize the system under test
  HostMemory outputOnHost = sut.runTest(inputData, outputData);
 
  //move data in local array
  outputOnHostPtr = outputOnHost.get<float>();
  for (size_t idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = outputOnHostPtr[idx] ;

  // Validate the returned data array
  for (size_t idx = 0; idx < sut.lengthOutputData / 2; ++idx)
    if (outputData[idx]  != 0)
    {
      cout << "encountered incorrect output on idx: " << idx << endl;
      cout << "expected: " << 0 << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData / 2);  //do not print salt
      //fail the current test

      CHECK(false);
      return;
    }
  
  // validate the output bound memory: it should al be the salt value
  for (size_t idx = sut.lengthOutputData / 2; idx < sut.lengthOutputData; ++idx)
    if (outputData[idx]  != salt )
    {
      cout << "encountered memory writes on invalid memory idx: " << idx << endl;
      cout << "expected: " << salt << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData);
      //fail the current test

      CHECK(false);
      return;
    }

  delete [] inputData;
  delete [] outputData;
}


TEST(CoherentNoComplex1SampleTest)
{
  // ***********************************************************
  // tests if the stokes parameters are calculate correctly. For a single sample
  // I = X *  con(X) + Y * con(Y)
  // Q = X *  con(X) - Y * con(Y)
  // U = 2 * RE(X * con(Y))
  // V = 2 * IM(X * con(Y))
  //
  // This reduces to (validate on paper by Wouter and John):
  // PX = RE(X) * RE(X) + IM(X) * IM(X)
  // PY = RE(Y) * RE(Y) + IM(Y) * IM(Y)
  // I = PX + PY
  // Q = PX - PY
  // 
  // U = 2 * (RE(X) * RE(Y) + IM(X) * IM(Y))
  // V = 2 * (IM(X) * RE(Y) - RE(X) * IM(Y))

  CoherentStokesTester sut(
    1,  //NR_CHANNELS
    1,  //NR_SAMPLES_PER_CHANNEL
    1,  //TIME_PARALLEL_FACTOR
    1,  //NR_TABS
    4,  //NR_COHERENT_STOKES 
    1); //INTEGRATION_SIZE
  float salt = 42.0f;
  // Get the input and output arrays
  // ************************************************************* 
  float * inputData = new float[sut.lengthInputData];
  float * outputData = new float[sut.lengthOutputData]; 
  float * outputOnHostPtr;

  //initialize
  for (unsigned idx = 0; idx < (sut.lengthInputData /2) / 2; ++idx)
  {
    inputData[idx * 2] = 2.0f;
    inputData[idx * 2 + 1] = 0.0f;
  }
  for (unsigned idx = (sut.lengthInputData /2) / 2; idx < (sut.lengthInputData /2); ++idx)
  {
    inputData[idx * 2] = 1.0f;
    inputData[idx * 2 + 1] = 0.0f;
  }
  for (unsigned idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = salt;

  // excersize the system under test
  HostMemory outputOnHost = sut.runTest(inputData, outputData);
 
  //move data in local array
  outputOnHostPtr = outputOnHost.get<float>();
  for (size_t idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = outputOnHostPtr[idx] ;

  // Validate the returned data array
  
  if (outputData[0] != 5.0f ||  // I = 2 ^ 2 + 1 ^2 
      outputData[1] != 3.0f ||  // Q = 2 ^ 2 - 1 ^2
      outputData[2] != 4.0f ||  // U = 2 * (2 * 1 + 0 * 0) 
      outputData[3] != 0.0f     // V = 2 * (0 * 1 - 2 * 0)
    )
    {
      cout << "encountered incorrect output." <<  endl;
      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData / 2);  //do not print salt
      //fail the current test

      CHECK(false);
      return;
    }
  
  // validate the output bound memory: it should al be the salt value
  for (size_t idx = sut.lengthOutputData / 2; idx < sut.lengthOutputData; ++idx)
    if (outputData[idx]  != salt )
    {
      cout << "encountered memory writes on invalid memory idx: " << idx << endl;
      cout << "expected: " << salt << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData);
      //fail the current test

      CHECK(false);
      return;
    }

  delete [] inputData;
  delete [] outputData;
}

TEST(CoherentComplex1SampleTest)
{
  // ***********************************************************
  // tests if the stokes parameters are calculate correctly. For a single sample
  // insert complex values
  // I = X *  con(X) + Y * con(Y)
  // Q = X *  con(X) - Y * con(Y)
  // U = 2 * RE(X * con(Y))
  // V = 2 * IM(X * con(Y))
  //
  // This reduces to (validate on paper by Wouter and John):
  // PX = RE(X) * RE(X) + IM(X) * IM(X)
  // PY = RE(Y) * RE(Y) + IM(Y) * IM(Y)
  // I = PX + PY
  // Q = PX - PY
  // 
  // U = 2 * (RE(X) * RE(Y) + IM(X) * IM(Y))
  // V = 2 * (IM(X) * RE(Y) - RE(X) * IM(Y))

  CoherentStokesTester sut(
    1,  //NR_CHANNELS
    1,  //NR_SAMPLES_PER_CHANNEL
    1,  //TIME_PARALLEL_FACTOR
    1,  //NR_TABS
    4,  //NR_COHERENT_STOKES 
    1); //INTEGRATION_SIZE
  float salt = 42.0f;
  // Get the input and output arrays
  // ************************************************************* 
  float * inputData = new float[sut.lengthInputData];
  float * outputData = new float[sut.lengthOutputData]; 
  float * outputOnHostPtr;

  //initialize
  for (unsigned idx = 0; idx < (sut.lengthInputData /2) / 2; ++idx)
  {
    inputData[idx * 2] = 0.0f;
    inputData[idx * 2 + 1] = 2.0f;
  }
  for (unsigned idx = (sut.lengthInputData /2) / 2; idx < (sut.lengthInputData /2); ++idx)
  {
    inputData[idx * 2] = 0.0f;
    inputData[idx * 2 + 1] = 1.0f;
  }
  for (unsigned idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = salt;

  // excersize the system under test
  HostMemory outputOnHost = sut.runTest(inputData, outputData);
 
  //move data in local array
  outputOnHostPtr = outputOnHost.get<float>();
  for (size_t idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = outputOnHostPtr[idx] ;

  // Validate the returned data array
  
  if (outputData[0] != 5.0f ||  // I = 2 ^ 2 + 1 ^ 2 
      outputData[1] != 3.0f ||  // Q = 2 ^ 2 - 1 ^ 2
      outputData[2] != 4.0f ||  // U = 2 * (2 * 1 + 0 * 0) 
      outputData[3] != 0.0f     // V = 2 * (0 * 1 - 2 * 0)
    )
    {
      cout << "encountered incorrect output." <<  endl;
      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData / 2);  //do not print salt
      //fail the current test

      CHECK(false);
      return;
    }
  
  // validate the output bound memory: it should al be the salt value
  for (size_t idx = sut.lengthOutputData / 2; idx < sut.lengthOutputData; ++idx)
    if (outputData[idx]  != salt )
    {
      cout << "encountered memory writes on invalid memory idx: " << idx << endl;
      cout << "expected: " << salt << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData);
      //fail the current test

      CHECK(false);
      return;
    }

  delete [] inputData;
  delete [] outputData;
}

TEST(Coherent4DifferentValuesSampleTest)
{
  // ***********************************************************
  // tests if the stokes parameters are calculate correctly. For a single sample
  // Insert both complex and non complex data
  // I = X *  con(X) + Y * con(Y)
  // Q = X *  con(X) - Y * con(Y)
  // U = 2 * RE(X * con(Y))
  // V = 2 * IM(X * con(Y))
  //
  // This reduces to (validate on paper by Wouter and John):
  // PX = RE(X) * RE(X) + IM(X) * IM(X)
  // PY = RE(Y) * RE(Y) + IM(Y) * IM(Y)
  // I = PX + PY
  // Q = PX - PY
  // 
  // U = 2 * (RE(X) * RE(Y) + IM(X) * IM(Y))
  // V = 2 * (IM(X) * RE(Y) - RE(X) * IM(Y))

  CoherentStokesTester sut(
    1,  //NR_CHANNELS
    1,  //NR_SAMPLES_PER_CHANNEL
    1,  //TIME_PARALLEL_FACTOR
    1,  //NR_TABS
    4,  //NR_COHERENT_STOKES 
    1); //INTEGRATION_SIZE
  float salt = 42.0f;
  // Get the input and output arrays
  // ************************************************************* 
  float * inputData = new float[sut.lengthInputData];
  float * outputData = new float[sut.lengthOutputData]; 
  float * outputOnHostPtr;

  //initialize
  for (unsigned idx = 0; idx < (sut.lengthInputData /2) / 2; ++idx)
  {
    inputData[idx * 2] = 1.0f;
    inputData[idx * 2 + 1] = 2.0f;
  }
  for (unsigned idx = (sut.lengthInputData /2) / 2; idx < (sut.lengthInputData /2); ++idx)
  {
    inputData[idx * 2] = 3.0f;
    inputData[idx * 2 + 1] = 4.0f;
  }
  for (unsigned idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = salt;

  // excersize the system under test
  HostMemory outputOnHost = sut.runTest(inputData, outputData);
 
  //move data in local array
  outputOnHostPtr = outputOnHost.get<float>();
  for (size_t idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = outputOnHostPtr[idx] ;

  // Validate the returned data array
  
  if (outputData[0] != 30.0f  ||  // I = (1 ^ 2 + 2 ^ 2) + (3 ^ 2 + 4 ^ 2) = 1 + 4 + 9 + 16 = 30
      outputData[1] != -20.0f ||  // Q = (1 ^ 2 + 2 ^ 2) - (3 ^ 2 + 4 ^ 2) = 1 + 4 - 9 - 16 = -20
      outputData[2] != 22.0f  ||  // U = 2 * (1 * 3 + 2 * 4) = 2 * (3 + 8) = 2 * 11 = 22
      outputData[3] != 4.0f       // V = 2 * (2 * 3 - 1 * 4) = 2 * (6 - 4) = 2 * 2 = 4
    )
    {
      cout << "encountered incorrect output." <<  endl;
      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData / 2);  //do not print salt
      //fail the current test

      CHECK(false);
      return;
    }
  
  // validate the output bound memory: it should al be the salt value
  for (size_t idx = sut.lengthOutputData / 2; idx < sut.lengthOutputData; ++idx)
    if (outputData[idx]  != salt )
    {
      cout << "encountered memory writes on invalid memory idx: " << idx << endl;
      cout << "expected: " << salt << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData);
      //fail the current test

      CHECK(false);
      return;
    }

  delete [] inputData;
  delete [] outputData;
}

TEST(BasicIntegrationTest)
{
  // ***********************************************************
  // Test if the integration works by inputting non complex ones 
  // and integrating over the total number of samples
  // This should result in 2 * num samples in both I and V
  unsigned NR_SAMPLES_PER_CHANNEL = 16;
  CoherentStokesTester sut(
    1,  //NR_CHANNELS
    NR_SAMPLES_PER_CHANNEL,  //NR_SAMPLES_PER_CHANNEL
    1,  //TIME_PARALLEL_FACTOR
    1,  //NR_TABS
    4,  //NR_COHERENT_STOKES 
    16); //INTEGRATION_SIZE
  float salt = 42.0f;
  // Get the input and output arrays
  // ************************************************************* 
  float * inputData = new float[sut.lengthInputData];
  float * outputData = new float[sut.lengthOutputData]; 
  float * outputOnHostPtr;

  //initialize
  for (unsigned idx = 0; idx < sut.lengthInputData/2; ++idx)
  {
    inputData[idx * 2] = 1.0f;
    inputData[idx * 2 + 1] = 0.0f;
  }
  for (unsigned idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = salt;

  // excersize the system under test
  HostMemory outputOnHost = sut.runTest(inputData, outputData);
 
  //move data in local array
  outputOnHostPtr = outputOnHost.get<float>();
  for (size_t idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = outputOnHostPtr[idx] ;

  // Validate the returned data array
  for (size_t idx = 0; idx < sut.lengthOutputData / 2; ++idx)
  {
    if (outputData[0] != 2.0f * NR_SAMPLES_PER_CHANNEL ||
        outputData[1] != 0.0f ||
        outputData[2] != 2.0f * NR_SAMPLES_PER_CHANNEL ||
        outputData[3] != 0.0f )
    {
      cout << "encountered incorrect output on idx: " << idx << endl;
      cout << "expected: " << 0 << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData / 2);  //do not print salt
      //fail the current test

      CHECK(false);
      return;
    }
  }
  // validate the output bound memory: it should al be the salt value
  for (size_t idx = sut.lengthOutputData / 2; idx < sut.lengthOutputData; ++idx)
    if (outputData[idx]  != salt )
    {
      cout << "encountered memory writes on invalid memory idx: " << idx << endl;
      cout << "expected: " << salt << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData);
      //fail the current test

      CHECK(false);
      return;
    }

  delete [] inputData;
  delete [] outputData;
}

TEST(Coherent2DifferentValuesAllDimTest)
{
  // ***********************************************************
  // Full test performing all functionalities and runtime validate that the output
  // is correct.
  // 1. Insert both complex and non complex values. This should result specific values
  // for all Stokes parameters
  // 2. Do it time parallel
  // 3. Integrate 
  // 4. Use tabs and channels
  size_t NR_CHANNELS = 16;
  size_t NR_SAMPLES_PER_CHANNEL = 1024;
  size_t TIME_PARALLEL_FACTOR = 4;
  size_t NR_TABS = 16;
  size_t NR_COHERENT_STOKES = 4;
  size_t INTEGRATION_SIZE = 4;

  
  CoherentStokesTester sut(
    NR_CHANNELS,
    NR_SAMPLES_PER_CHANNEL, 
    TIME_PARALLEL_FACTOR, 
    NR_TABS,  
    NR_COHERENT_STOKES, 
    INTEGRATION_SIZE); 
  
  float salt = 42.0f;
  // Get the input and output arrays
  // ************************************************************* 
  float * inputData = new float[sut.lengthInputData];
  float * outputData = new float[sut.lengthOutputData]; 
  float * outputOnHostPtr;


  //initialize
  for (unsigned idx = 0; idx < (sut.lengthInputData /2) ; ++idx)
  {
    inputData[idx * 2] = 1.0f;
    inputData[idx * 2 + 1] = 2.0f;
  }
  
  for (unsigned idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = salt;

  // excersize the system under test
  HostMemory outputOnHost = sut.runTest(inputData, outputData);
 
  //move data in local array
  outputOnHostPtr = outputOnHost.get<float>();
  for (size_t idx = 0; idx < sut.lengthOutputData; ++idx)
    outputData[idx] = outputOnHostPtr[idx] ;

  // Validate the returned data array
  // The number of repeats of the same value is depending on sample channels and integrations size
  size_t value_repeat = NR_CHANNELS * (NR_SAMPLES_PER_CHANNEL / INTEGRATION_SIZE);
  // For stokes parameters
  size_t size_tab = value_repeat * 4;
  for (size_t idx_tab = 0; idx_tab < NR_TABS; ++idx_tab)
  {
    // I
    for (size_t idx_value_repeat = 0 ; idx_value_repeat < value_repeat; ++idx_value_repeat)
    {
      if(outputData[idx_tab * size_tab + idx_value_repeat] != 10 * INTEGRATION_SIZE ||
         outputData[idx_tab * size_tab + value_repeat + idx_value_repeat] != 0 ||
         outputData[idx_tab * size_tab + value_repeat * 2 + idx_value_repeat] != 10 * INTEGRATION_SIZE ||
         outputData[idx_tab * size_tab + value_repeat * 3 +idx_value_repeat] != 0
        )
      {
         cout << "encountered incorrect output. on stokes parameter q, idx base: " <<
           idx_tab * size_tab + idx_value_repeat << endl;
            
         CHECK(false);
         return;
      }
    }
  }
 

  // validate the output bound memory: it should al be the salt value
  for (size_t idx = sut.lengthOutputData / 2; idx < sut.lengthOutputData; ++idx)
    if (outputData[idx]  != salt )
    {
      cout << "encountered memory writes on invalid memory idx: " << idx << endl;
      cout << "expected: " << salt << " received: "  << outputData[idx] << endl;

      //pretty print the input and output data
      exit_with_print(inputData, outputData,
                       sut.lengthInputData, sut.lengthOutputData);
      //fail the current test

      CHECK(false);
      return;
    }

  delete [] inputData;
  delete [] outputData;
}

Exception::TerminateHandler t(Exception::terminate);

int main()
{

  INIT_LOGGER("tCoherentStokes");
  
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
  

  int exitStatus = UnitTest::RunAllTests();
  return exitStatus > 0 ? 1 : 0;
}


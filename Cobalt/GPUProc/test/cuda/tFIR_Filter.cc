//# FIR_Filter.cc
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
//# $Id: tFIR_Filter.cc 26571 2013-09-16 18:18:19Z amesfoort $

#include <lofar_config.h>

#define COMPLEX 2       // do not change

#define NR_BITS_PER_SAMPLE 8
#define NR_POLARIZATIONS         2 
#define NR_TAPS                 16

#define NR_SUBBANDS 1
#define NR_STATIONS 20
#define NR_SAMPLES_PER_CHANNEL 128
#define NR_CHANNELS 64

#if NR_BITS_PER_SAMPLE == 16
#error unsupported by this test at the moment
typedef signed short SampleType;
#elif NR_BITS_PER_SAMPLE == 8
typedef signed char SampleType;
#else
#error unsupported NR_BITS_PER_SAMPLE
#endif

#include <cstdlib> 
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <boost/lexical_cast.hpp>

#include <Common/LofarLogger.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/FilterBank.h>

#include "../TestUtil.h"
#include "../fpequals.h"

using namespace std;
using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace LOFAR::Cobalt::gpu;

using boost::lexical_cast;

int test()
{
  bool testOk = true;

  string kernelFile = "FIR_Filter.cu";
  string function = "FIR_filter";

  // Get an instantiation of the default parameters
  CompileDefinitions definitions = CompileDefinitions();
  CompileFlags flags = defaultCompileFlags();

  // ****************************************
  // Compile to ptx
  // Set op string string pairs to be provided to the compiler as defines
  definitions["NR_TAPS"] = lexical_cast<string>(NR_TAPS);
  definitions["NR_STABS"] = lexical_cast<string>(NR_STATIONS);
  definitions["NR_CHANNELS"] = lexical_cast<string>(NR_CHANNELS);
  definitions["NR_SAMPLES_PER_CHANNEL"] = 
    lexical_cast<string>(NR_SAMPLES_PER_CHANNEL);
  definitions["NR_POLARIZATIONS"] = lexical_cast<string>(NR_POLARIZATIONS);
  definitions["COMPLEX"] = lexical_cast<string>(COMPLEX);
  definitions["NR_BITS_PER_SAMPLE"] = lexical_cast<string>(NR_BITS_PER_SAMPLE);
  definitions["NR_SUBBANDS"] = lexical_cast<string>(NR_SUBBANDS);

  // Create a default context
  Platform pf;
  Device device(0);
  Context ctx(device);
  Stream stream(ctx);
  vector<Device> devices(1, ctx.getDevice());
  string ptx = createPTX(kernelFile, definitions, flags, devices);
  Module module(createModule(ctx, kernelFile, ptx));
  Function hKernel(module, function);

  // Create the needed data
  unsigned sizeFilteredData = 
    NR_STATIONS * NR_POLARIZATIONS * NR_SAMPLES_PER_CHANNEL * 
    NR_CHANNELS * COMPLEX;
  HostMemory rawFilteredData = 
    getInitializedArray(ctx, sizeFilteredData * sizeof(float), 0.0f);

  unsigned sizeSampledData = 
    NR_STATIONS * NR_SAMPLES_PER_CHANNEL * NR_CHANNELS * 
    NR_POLARIZATIONS * COMPLEX;
  HostMemory rawInputSamples = 
    getInitializedArray(ctx, sizeSampledData * sizeof(signed char), char(0));

  unsigned sizeWeightsData = NR_CHANNELS * NR_TAPS;
  HostMemory rawFirWeights = 
    getInitializedArray(ctx, sizeWeightsData * sizeof(float), 0.0f);

  unsigned sizeHistoryData = 
    NR_SUBBANDS * NR_STATIONS * (NR_TAPS - 1) * NR_CHANNELS * NR_POLARIZATIONS * COMPLEX;
  HostMemory rawHistoryData = 
    getInitializedArray(ctx, sizeHistoryData * sizeof(signed char), char(0));

  // Data on the gpu
  DeviceMemory devFilteredData(ctx, sizeFilteredData * sizeof(float));
  DeviceMemory devSampledData(ctx, sizeSampledData * sizeof(signed char));
  DeviceMemory devFirWeights(ctx, sizeWeightsData * sizeof(float));
  DeviceMemory devHistoryData(ctx, sizeHistoryData * sizeof(signed char));

  unsigned station, sample, ch, pol;

  // Calculate the number of threads in total and per block

  // Dit moet nog opgevraagd worden en niet als magisch getal
  int MAXNRCUDATHREADS = 64;

  size_t maxNrThreads = MAXNRCUDATHREADS;
  unsigned totalNrThreads = NR_CHANNELS * NR_POLARIZATIONS * 2;
  unsigned nrPasses = (totalNrThreads + maxNrThreads - 1) / maxNrThreads;
  
  Grid globalWorkSize(nrPasses, NR_STATIONS); 
  Block localWorkSize(totalNrThreads / nrPasses, 1); 

  MultiDimArray<signed char, 5>
    inputSamplesArr(boost::extents
                    [NR_STATIONS]
                    [NR_SAMPLES_PER_CHANNEL]
                    [NR_CHANNELS]
                    [NR_POLARIZATIONS]
                    [COMPLEX],
                    rawInputSamples.get<signed char>(), false);

  MultiDimArray<float, 2> 
    firWeightsArr(boost::extents
                  [NR_CHANNELS]
                  [NR_TAPS],
                  rawFirWeights.get<float>(), false);

  MultiDimArray<float, 5>
    filteredDataArr(boost::extents
                    [NR_STATIONS]
                    [NR_POLARIZATIONS]
                    [NR_SAMPLES_PER_CHANNEL]
                    [NR_CHANNELS]
                    [COMPLEX],
                    rawFilteredData.get<float>(), false);

  MultiDimArray<signed char, 6>
    historyDataArr(boost::extents
                    [NR_SUBBANDS]
                    [NR_STATIONS]
                    [NR_TAPS - 1]
                    [NR_CHANNELS]
                    [NR_POLARIZATIONS]
                    [COMPLEX],
                    rawHistoryData.get<signed char>(), false);

  // Test 1: Single impulse test on single non-zero weight
  station = ch = pol = 0;
  sample = 0;
  rawFirWeights.get<float>()[0] = 2.0f;
  inputSamplesArr[station][sample][ch][pol][0] = 3;

  // Copy input vectors from host memory to GPU buffers.
  stream.writeBuffer(devFilteredData, rawFilteredData, true);
  stream.writeBuffer(devSampledData, rawInputSamples, true);
  stream.writeBuffer(devFirWeights, rawFirWeights, true);
  stream.writeBuffer(devHistoryData, rawHistoryData, true);

  // Run the kernel on the created data
  hKernel.setArg(0, devFilteredData);
  hKernel.setArg(1, devSampledData);
  hKernel.setArg(2, devFirWeights);
  hKernel.setArg(3, devHistoryData);
  size_t subbandIdx = 0;
  hKernel.setArg(4, subbandIdx);

  // Run the kernel
  stream.synchronize();
  stream.launchKernel(hKernel, globalWorkSize, localWorkSize);
  stream.synchronize();

  stream.readBuffer(rawFilteredData, devFilteredData, true);

  // Expected output: St0, pol0, ch0, sampl0: 6. The rest all 0.
  // However, in modes other than 16 bit mode, all gains are scaled to match 16 bit mode.
  unsigned scale = 1;
  if (NR_BITS_PER_SAMPLE != 16)
    scale = 16;
  if (rawFilteredData.get<float>()[0] != 6.0f * scale) 
  {
    cerr << "FIR_FilterTest 1: Expected at idx 0: " << 6 * scale 
              << "; got: " << rawFilteredData.get<float>()[0] << endl;

    testOk = false;
  }
  cerr << "Weights returned " << rawFilteredData.get<float>()[0] 
            << endl;

  const size_t nrExpectedZeros = sizeFilteredData - 1;
  size_t nrZeros = 0;
  for (unsigned i = 1; i < sizeFilteredData; i++) 
    if (rawFilteredData.get<float>()[i] == 0.0f) {
      nrZeros ++;
    } else {
      cerr << "filteredData[" << i << "] = " << rawFilteredData.get<float>()[i]
           << endl;
    }

  if (nrZeros != nrExpectedZeros) 
  {
    cerr << "FIR_FilterTest 1: Unexpected non-zero(s). Only " << nrZeros 
         << " zeros out of " << nrExpectedZeros << endl;
    testOk = false;
  }


  // Test 2: Impulse train 2*NR_TAPS apart. All st, all ch, all pol.
  for (ch = 0; ch <NR_CHANNELS; ch++) {
    for (unsigned tap = 0; tap < NR_TAPS; tap++) {
      firWeightsArr[ch][tap] = ch + tap;
    }
  }

  for (station = 0; station < NR_STATIONS; station++) {
    for (sample = 0; sample < NR_SAMPLES_PER_CHANNEL; sample += 2 * NR_TAPS) {
      for (ch = 0; ch <NR_CHANNELS; ch++) {
        for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
          inputSamplesArr[station][sample][ch][pol][0] = station;
        }
      }
    }
  }

  // Copy input vectors from host memory to GPU buffers.
  stream.writeBuffer(devFilteredData, rawFilteredData, true);
  stream.writeBuffer(devSampledData, rawInputSamples, true);
  stream.writeBuffer(devFirWeights, rawFirWeights, true);
  stream.writeBuffer(devHistoryData, rawHistoryData, true);

  // Run the kernel
  stream.synchronize();
  stream.launchKernel(hKernel, globalWorkSize, localWorkSize);
  stream.synchronize();

  stream.readBuffer(rawFilteredData, devFilteredData, true);

  // Expected output: sequences of (filterbank scaled by station nr, NR_TAPS
  // zeros)
  unsigned nrErrors = 0;
  for (station = 0; station < NR_STATIONS; station++) {
    for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
      unsigned s;
      for (sample = 0; sample < NR_SAMPLES_PER_CHANNEL / (2 * NR_TAPS); 
           sample += s) {
        for (s = 0; s < NR_TAPS; s++) {
          for (ch = 0; ch <NR_CHANNELS; ch++) {
            if (filteredDataArr[station][pol][sample + s][ch][0] != 
                scale * station * firWeightsArr[ch][s]) {
              if (++nrErrors < 100) { // limit spam
                cerr << "2a.filtered[" << station << "][" << pol << "][" 
                     << sample+s << "][" << ch << "][0] (sample=" << sample
                     << " s="<<s<<") = " << setprecision(9+1)
                     << filteredDataArr[station][pol][sample + s][ch][0]
                     << endl;
              }
            }
            if (filteredDataArr[station][pol][sample + s][ch][1] != 0.0f) {
              if (++nrErrors < 100) {
                cerr << "2a imag non-zero: " << setprecision(9+1)
                     << filteredDataArr[station][pol][sample + s][ch][1]
                     << endl;
              }
            }
          }
        }

        for ( ; s < 2 * NR_TAPS; s++) {
          for (ch = 0; ch < NR_CHANNELS; ch++) {
            if (filteredDataArr[station][pol][sample + s][ch][0] != 0.0f ||
                filteredDataArr[station][pol][sample + s][ch][1] != 0.0f) {
              if (++nrErrors < 100) {
                cerr << "2b.filtered[" << station << "][" << pol << "][" 
                     << sample+s << "][" << ch << "][0] (sample=" << sample
                     << " s=" << s <<") = " << setprecision(9+1)
                     << filteredDataArr[station][pol][sample + s][ch][0]
                     << ", "<< filteredDataArr[station][pol][sample + s][ch][1]
                     << endl;
              }
            }
          }
        }
      }
    }
  }
  if (nrErrors == 0) {
    cout << "FIR_FilterTest 2: test OK" << endl;
  } else {
    cerr << "FIR_FilterTest 2: " << nrErrors << " unexpected output values"
         << endl;
    testOk = false;
  }


  // Test 3: Scaled step test (scaled DC gain) on KAISER filterbank. Non-zero
  // imag input.
  FilterBank filterBank(true, NR_TAPS, NR_CHANNELS, KAISER);
  filterBank.negateWeights(); // not needed for testing, but as we use it
  //filterBank.printWeights();

  assert(firWeightsArr.num_elements() == 
         filterBank.getWeights().num_elements());

  double* expectedSums = new double[NR_CHANNELS];
  memset(expectedSums, 0, NR_CHANNELS * sizeof(double));
  for (ch = 0; ch < NR_CHANNELS; ch++) {
    for (unsigned tap = 0; tap < NR_TAPS; tap++) {
      firWeightsArr[ch][tap] = filterBank.getWeights()[ch][tap];
      expectedSums[ch] += firWeightsArr[ch][tap];
    }
  }

  for (station = 0; station < NR_STATIONS; station++) {
    for (sample = 0; sample < NR_SAMPLES_PER_CHANNEL; sample++) {
      for (ch = 0; ch < NR_CHANNELS; ch++) {
        for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
          inputSamplesArr[station][sample][ch][pol][0] = 2; // real
          inputSamplesArr[station][sample][ch][pol][1] = 3; // imag
        }
      }
    }
  }

  // Copy input vectors from host memory to GPU buffers.
  stream.writeBuffer(devFilteredData, rawFilteredData, true);
  stream.writeBuffer(devSampledData, rawInputSamples, true);
  stream.writeBuffer(devFirWeights, rawFirWeights, true);
  stream.writeBuffer(devHistoryData, rawHistoryData, true);

  // Run the kernel
  stream.synchronize();
  stream.launchKernel(hKernel, globalWorkSize, localWorkSize);
  stream.synchronize();

  stream.readBuffer(rawFilteredData, devFilteredData, true);

  nrErrors = 0;
  const float eps = 4.0f * numeric_limits<float>::epsilon();
  for (station = 0; station < NR_STATIONS; station++) {
    for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
      // Skip first NR_TAPS - 1 samples; they contain the filtered history data
      for (sample = NR_TAPS - 1; sample < NR_SAMPLES_PER_CHANNEL; sample++) {
        for (ch = 0; ch < NR_CHANNELS; ch++) {
          // Expected sum must also be scaled by 2 and 3
          if (!fpEquals(filteredDataArr[station][pol][sample][ch][0], 
                        (float)(2 * scale * expectedSums[ch]), eps)) {
            if (++nrErrors < 100) { // limit spam
              cerr << "3a.filtered[" << station << "][" << pol << "]["
                   << sample << "][" << ch << "][0] = " << setprecision(9+1)
                   << filteredDataArr[station][pol][sample][ch][0] 
                   << " 2*weight = " << 2 * scale * expectedSums[ch] << endl;
            }
          }
          if (!fpEquals(filteredDataArr[station][pol][sample][ch][1], 
                        (float)(3 * scale * expectedSums[ch]), eps)) {
            if (++nrErrors < 100) {
              cerr << "3b.filtered[" << station << "][" << pol << "][" 
                   << sample << "][" << ch << "][1] = " << setprecision(9+1)
                   << filteredDataArr[station][pol][sample][ch][1]
                   << " 3*weight = " << 3 * scale * expectedSums[ch] << endl;
            }
          }
        }
      }
    }
  }
  delete[] expectedSums;
  if (nrErrors == 0) {
    cout << "FIR_FilterTest 3: test OK" << endl;
  } else {
    cerr << "FIR_FilterTest 3: " << nrErrors << " unexpected output values"
         << endl;
    testOk = false;
  }


  // Test 4: Test the use of history samples, by invoking the GPU kernel more
  // than once.

  // Clear all global memory blocks first.
  memset(inputSamplesArr.origin(), 0,
         inputSamplesArr.num_elements() * sizeof(signed char));
  memset(firWeightsArr.origin(), 0, 
         firWeightsArr.num_elements() * sizeof(float));
  memset(filteredDataArr.origin(), 0,
         filteredDataArr.num_elements() * sizeof(float));
  memset(historyDataArr.origin(), 0,
         historyDataArr.num_elements() * sizeof(signed char));

  // Set FIR filter weights: a triangle shaped pulse response.
  for (ch = 0; ch < NR_CHANNELS; ch++) {
    for (unsigned tap = 0; tap < NR_TAPS; tap++) {
      firWeightsArr[ch][tap] = (NR_TAPS - tap) / 16.0f;
    }
  }

  size_t halfNrTaps = NR_TAPS / 2;
  size_t quartNrTaps = halfNrTaps / 2;

  // Set input samples: a train of alternating real and imaginary pulses,
  // NR_TAPS / 4 samples apart.
  size_t n0, n1, n2, n3;
  for (station = 0; station < NR_STATIONS; station++) {
    for (sample = 0; sample < NR_SAMPLES_PER_CHANNEL; sample += NR_TAPS) {
      n0 = sample; 
      n1 = n0 + quartNrTaps;
      n2 = n1 + quartNrTaps;
      n3 = n2 + quartNrTaps;
      for (ch = 0; ch < NR_CHANNELS; ch++) {
        for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
          inputSamplesArr[station][n0][ch][pol][0] = 1.0f;
          inputSamplesArr[station][n1][ch][pol][1] = 1.0f;
          inputSamplesArr[station][n2][ch][pol][0] = -1.0f;
          inputSamplesArr[station][n3][ch][pol][1] = -1.0f;
        }
      }
    }
  }

  // Copy input vectors from host memory to GPU buffers.
  stream.writeBuffer(devFilteredData, rawFilteredData, true);
  stream.writeBuffer(devSampledData, rawInputSamples, true);
  stream.writeBuffer(devFirWeights, rawFirWeights, true);
  stream.writeBuffer(devHistoryData, rawHistoryData, true);

  // Run the kernel
  stream.synchronize();
  stream.launchKernel(hKernel, globalWorkSize, localWorkSize);
  stream.synchronize();

  stream.readBuffer(rawFilteredData, devFilteredData, true);

  // Verify output. This first NR_TAPS outputs represent the transient behaviour
  // of the FIR filter; the remaining output the steady state.
  nrErrors = 0;
  for (station = 0; station < NR_STATIONS; station++) {
    for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
      // transient part
      for (sample = 0; sample < halfNrTaps ; sample++) {
        for (ch = 0; ch < NR_CHANNELS; ch++) {
          if (filteredDataArr[station][pol][sample][ch][0] != 
              firWeightsArr[ch][sample] * scale) {
            nrErrors++;
            cerr << "4a.filtered[" << station << "][" << pol << "]["
                 << sample << "][" << ch << "][0] = "
                 << filteredDataArr[station][pol][sample][ch][0] << " != "
                 << firWeightsArr[ch][sample] * scale << endl;
          }
          if (filteredDataArr[station][pol][sample + quartNrTaps][ch][1] != 
              firWeightsArr[ch][sample] * scale) {
            nrErrors++;
            cerr << "4a.filtered[" << station << "][" << pol << "]["
                 << sample + quartNrTaps << "][" << ch << "][1] = "
                 << filteredDataArr[station][pol][sample + quartNrTaps][ch][1]
                 << " != " << firWeightsArr[ch][sample] * scale << endl;
          }
        }
      }
      // steady state part
      int sign = 1;
      for (sample = halfNrTaps; sample < NR_SAMPLES_PER_CHANNEL - quartNrTaps; 
           sample++) {
        sign = (sample % halfNrTaps == 0) ? -sign : sign;
        for (ch = 0; ch < NR_CHANNELS; ch++) {
          if (filteredDataArr[station][pol][sample][ch][0] !=
              firWeightsArr[ch][0] * scale * sign / 2) {
            nrErrors++;
            cerr << "4a.filtered[" << station << "][" << pol << "]["
                 << sample << "][" << ch << "][0] = "
                 << filteredDataArr[station][pol][sample][ch][0] << " != "
                 << firWeightsArr[ch][0] * scale * sign / 2 << endl;
          }
          if (filteredDataArr[station][pol][sample + quartNrTaps][ch][1] !=
              firWeightsArr[ch][0] * scale * sign / 2) {
            nrErrors++;
            cerr << "4a.filtered[" << station << "][" << pol << "]["
                 << sample + quartNrTaps << "][" << ch << "][1] = "
                 << filteredDataArr[station][pol][sample + quartNrTaps][ch][1] 
                 << " != " << firWeightsArr[ch][0] * scale * sign / 2 << endl;
          }
        }
      }
    }
  }

  // Run the kernel for the second time. This time the history buffer should be
  // filled with previous samples. Hence we should NOT see the transient
  // behaviour of the FIR filter we saw previously in the first run.
  stream.synchronize();
  stream.launchKernel(hKernel, globalWorkSize, localWorkSize);
  stream.synchronize();

  stream.readBuffer(rawFilteredData, devFilteredData, true);

  // Verify output. This time, there should be no transient behaviour.
  for (station = 0; station < NR_STATIONS; station++) {
    for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
      int sign = -1;
      for (sample = 0; sample < NR_SAMPLES_PER_CHANNEL - quartNrTaps; sample++) {
        sign = (sample % halfNrTaps == 0) ? -sign : sign;
        for (ch = 0; ch < NR_CHANNELS; ch++) {
          if (filteredDataArr[station][pol][sample][ch][0] !=
              firWeightsArr[ch][0] * scale * sign / 2) {
            nrErrors++;
            cerr << "4b.filtered[" << station << "][" << pol << "]["
                 << sample << "][" << ch << "][0] = "
                 << filteredDataArr[station][pol][sample][ch][0] << " != "
                 << firWeightsArr[ch][0] * scale * sign / 2 << endl;
          }
          if (filteredDataArr[station][pol][sample + quartNrTaps][ch][1] !=
              firWeightsArr[ch][0] * scale * sign / 2) {
            nrErrors++;
            cerr << "4b.filtered[" << station << "][" << pol << "]["
                 << sample + quartNrTaps << "][" << ch << "][1] = "
                 << filteredDataArr[station][pol][sample + quartNrTaps][ch][1]
                 << " != " << firWeightsArr[ch][0] * scale * sign / 2 << endl;
          }
        }
      }
    }
  }

  if (nrErrors == 0) {
    cout << "FIR_FilterTest 4: test OK" << endl;
  } else {
    cerr << "FIR_FilterTest 4: " << nrErrors << " unexpected output values"
         << endl;
    testOk = false;
  }

  return testOk ? 0 : 1;
}

int main()
{
  INIT_LOGGER("tFIR_Filter");
  try {
    gpu::Platform pf;
  } catch (gpu::GPUException&) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 3;
  }
  return test() > 0;
}


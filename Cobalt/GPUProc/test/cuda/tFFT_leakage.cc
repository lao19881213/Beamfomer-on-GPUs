//# tFFT_leakage.cc
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
//# $Id: tFFT_leakage.cc 26819 2013-10-02 12:02:44Z amesfoort $

#include <lofar_config.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <fftw3.h>

#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/Kernels/FFT_Kernel.h>
#include <GPUProc/Kernels/FIR_FilterKernel.h>

#include <CoInterface/Parset.h>
#include <GPUProc/FilterBank.h>
#include <GPUProc/SubbandProcs/CorrelatorSubbandProc.h>
#include <GPUProc/cuda/Pipelines/Pipeline.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/gpu_wrapper.h>

using namespace std;
using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace LOFAR::Cobalt::gpu;

size_t nrErrors = 0;

const float EPSILON = 1e-5;

template<typename T> T sqr(const T &x)
{
  return x * x;
}

float amp(const float x[2])
{
  return sqrt(sqr(x[0]) + sqr(x[1]));
}

float amp(const fcomplex &x)
{
  return sqrt(sqr(x.real()) + sqr(x.imag()));
}

bool cmp_fcomplex(const fcomplex &a, const fcomplex &b, const float epsilon = EPSILON)
{
  const float absa = a.real() * a.real() + a.imag() * a.imag();
  const float absb = b.real() * b.real() + b.imag() * b.imag();

  return fabs(absa - absb) <= epsilon * epsilon;
}



int main() {
  INIT_LOGGER("tFFT_leakage");
  Parset ps("tFFT_leakage.in_.parset");

  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    return 3;
  }
  gpu::Device device(0);
  gpu::Context ctx(device);

  gpu::Stream stream(ctx);

  const size_t size = 128 * 1024;
  const int fftSize = 256;
  const unsigned nrFFTs = size / fftSize;

  // GPU buffers and plans
  gpu::HostMemory inout(ctx, ps.nrStations() * NR_POLARIZATIONS * ps.nrSamplesPerSubband()   * sizeof(fcomplex));
  gpu::DeviceMemory d_inout(ctx, size  * sizeof(fcomplex));



#define NR_POLARIZATIONS 2

  SubbandProcInputData::DeviceBuffers devInput(ps.nrBeams(),
    ps.nrStations(),
    NR_POLARIZATIONS,
    ps.nrSamplesPerSubband(),
    ps.nrBytesPerComplexSample(),
    ctx,
    // reserve enough space in inputSamples for the output of
    // the delayAndBandPassKernel.
    ps.nrStations() * NR_POLARIZATIONS * ps.nrSamplesPerSubband() * sizeof(std::complex<float>));

  DeviceMemory devFilteredData(ctx,
    // reserve enough space for the output of the
    // firFilterKernel,
    std::max(ps.nrStations() * NR_POLARIZATIONS * ps.nrSamplesPerSubband() * sizeof(std::complex<float>),
    // and the correlatorKernel.
    ps.nrBaselines() * ps.nrChannelsPerSubband() * NR_POLARIZATIONS * NR_POLARIZATIONS * sizeof(std::complex<float>)));

  // Create the FIR Filter weights
  DeviceMemory devFIRweights(ctx,
                             ps.nrChannelsPerSubband() * NR_TAPS * sizeof(float));
  FilterBank filterBank(true, NR_TAPS, ps.nrChannelsPerSubband(), KAISER);
  size_t fbBytes = filterBank.getWeights().num_elements() * sizeof(float);
  HostMemory fbBuffer(ctx, fbBytes); 
  // Copy to the hostbuffer
  std::memcpy(fbBuffer.get<void>(), filterBank.getWeights().origin(), fbBytes);


  HostMemory outFiltered(ctx, ps.nrStations() * NR_POLARIZATIONS * ps.nrSamplesPerSubband() * sizeof(std::complex<float>));
 
  std::vector<gpu::Device> devices(1,device);
  flags_type flags(defaultFlags());
  CompileDefinitions definitions(Kernel::compileDefinitions(ps));
  string ptx(createPTX(devices, "FIR_Filter.cu", flags, definitions));
  gpu::Module program =  createModule(ctx, "FIR_Filter.cu", ptx);

  FIR_FilterKernel firFilterKernel(ps, 
    stream, 
    program, 
    devFilteredData, 
    devInput.inputSamples,
    devFIRweights);

  FFT_Kernel fftFwdKernel(ctx, fftSize, nrFFTs, true, devFilteredData);

  fstream amplitudes("amplitudes.output",  std::fstream::out);
  double freq_begin = 4.0;
  double freq_end = 5.0;
  double freq_steps = 37.0;
  amplitudes << "freq_begin," << "freq_end," << "freq_steps," << "fftSize" << endl;
  amplitudes << freq_begin << ","<< freq_end << "," << freq_steps << "," << fftSize << endl;
  MultiDimArrayHostBuffer<i16complex,2 > inputDataRaw(
    boost::extents[ps.nrSamplesPerSubband()][2], ctx);

  // Initialize the input data with a sinus
  for( double freq = freq_begin; freq <= freq_end; freq += 1.0/freq_steps) //dus niet 128 hz  // eenheid in fft window widths
  {
    amplitudes << "Frequency: ," ;
    cerr << "Frequency: " << freq << endl;

    const double amplitude = 32767.0;

    // init buffers
    for (size_t i = 0; i < ps.nrSamplesPerSubband(); i++) {
      const double phase = (double)i * 2.0 * M_PI * freq / fftSize;
      const double real = amplitude * cos(phase);
      const double imag = amplitude * sin(phase);
      //inout.get<fcomplex>()[i] = fcomplex(real, imag);
      inputDataRaw[i][0] = i16complex((int16)rint(real),(int16)rint(imag));
      inputDataRaw[i][1] = i16complex((int16)rint(real),(int16)rint(imag));      
    }


    stream.writeBuffer(devFIRweights, fbBuffer, true);
    stream.writeBuffer(devInput.inputSamples, inputDataRaw, true);
    firFilterKernel.enqueue(stream);
    
    // As a test return the intermediate results
    //outFiltered.deviceToHost(true);
    //stream.synchronize(); 

    fftFwdKernel.enqueue(stream);
    stream.synchronize();
    stream.readBuffer(outFiltered, devFilteredData, true);


    // verify output -- we can't verify per-element because the leakage will
    // differ in pattern. So we collect the total signal leak outside our peak.
    double leak_gpu = 0.0;
    // Check for our frequency response
    for (int i = 0; i < fftSize; i++) 
    {
      amplitudes << amp(outFiltered.get<fcomplex>()[i]) << ",";
      if (i == (int)floor(freq) || i == (int)ceil(freq)) 
      {
        cerr << "GPU  signal peak @ freq " << i << ": " << amp(outFiltered.get<fcomplex>()[i]) << endl;
      } 
      else
      {

        leak_gpu += amp(outFiltered.get<fcomplex>()[i]);

      }
    }
    amplitudes << endl;
    cerr << "GPU  signal leak: " << leak_gpu << endl;
  }

  if (nrErrors > 0)
  {
    cerr << "Error: " << nrErrors << " unexpected output values" << endl;
  }

  return nrErrors > 0;
}


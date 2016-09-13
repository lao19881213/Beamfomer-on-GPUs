//# tFFT.cc: test the FFT kernel
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
//# $Id: tFFT.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <fftw3.h>

#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>
#include <CoInterface/BlockID.h>
#include <GPUProc/Kernels/FFT_Kernel.h>
#include <GPUProc/PerformanceCounter.h>

using namespace std;
using namespace LOFAR;
using namespace LOFAR::Cobalt;

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
  INIT_LOGGER("tFFT");

  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 0;
  }
  gpu::Device device(0);
  gpu::Context ctx(device);

  gpu::Stream stream(ctx);

  const size_t size = 128 * 1024;
  const int fftSize = 256;
  const unsigned nrFFTs = size / fftSize;

  // GPU buffers and plans
  gpu::HostMemory inout(ctx, size  * sizeof(fcomplex));
  gpu::DeviceMemory d_inout(ctx, size  * sizeof(fcomplex));

  // Dummy Block-ID
  BlockID blockId;

  FFT_Kernel fftFwdKernel(stream, fftSize, nrFFTs, true, d_inout);
  FFT_Kernel fftBwdKernel(stream, fftSize, nrFFTs, false, d_inout);

  // FFTW buffers and plans
  ASSERT(fftw_init_threads() != 0);
  fftw_plan_with_nthreads(4); // use up to 4 threads (don't care about test performance, but be impatient anyway...)

  fftwf_complex *f_inout = (fftwf_complex*)fftw_malloc(fftSize * nrFFTs * sizeof(fftw_complex));
  ASSERT(f_inout);

  fftwf_plan f_fftFwdPlan = fftwf_plan_many_dft(1, &fftSize, nrFFTs, // int rank, const int *n (=dims), int howmany,
                                      f_inout, NULL, 1, fftSize,    // fftw_complex *in, const int *inembed, int istride, int idist,
                                      f_inout, NULL, 1, fftSize,    // fftw_complex *out, const int *onembed, int ostride, int odist,
                                      FFTW_FORWARD, FFTW_ESTIMATE); // int sign, unsigned flags
  ASSERT(f_fftFwdPlan);

  // First run two very basic tests, then a third test with many transforms with FFTW output as reference.
  // All tests run complex-to-complex float, in-place.

  // *****************************************
  // Test 1: Impulse at origin
  //  Test correct usage of the runtime stat class functionality
  // *****************************************
  {

    // init buffers
    for (size_t i = 0; i < size; i++) {
      inout.get<fcomplex>()[i] = fcomplex(0.0f, 0.0f);
    }

    inout.get<fcomplex>()[0] = fcomplex(1.0f, 0.0f);

    // Forward FFT: compute and I/O
    stream.writeBuffer(d_inout, inout);
        
    fftFwdKernel.enqueue(blockId);
    stream.readBuffer(inout, d_inout, true);
    stream.synchronize();
    // do a call to the stats functionality 


    // verify output

    // Check for constant function in transfer domain. All real values 1.0 (like fftw, scaled). All imag must be 0.0.
    for (int i = 0; i < fftSize; i++) {
      if (inout.get<fcomplex>()[i] != fcomplex(1.0f, 0.0f)) {
        if (++nrErrors < 100) { // limit spam
          cerr << "fwd: " << i << ':' << inout.get<fcomplex>()[i] << endl;
        }
      }
    }

    // Backward FFT: compute and I/O
    fftFwdKernel.enqueue(blockId);
    stream.synchronize();
    stream.readBuffer(inout, d_inout, true);

    // See if we got only our scaled impuls back.
    if (inout.get<fcomplex>()[0] != fcomplex((float)fftSize, 0.0f)) {
      nrErrors += 1;
      cerr << "bwd: " << inout.get<fcomplex>()[0] << " at idx 0 should have been " << fcomplex((float)fftSize, 0.0f) << endl;
    }

    for (int i = 1; i < fftSize; i++) {
      if (inout.get<fcomplex>()[i] != fcomplex(0.0f, 0.0f)) {
        if (++nrErrors < 100) {
          cerr << "bwd: " << i << ':' << inout.get<fcomplex>()[i] << endl;
        }
      }
    }
  }

  // *****************************************
  // Test 2: Sine waves, compare signal leakage to fftw
  // *****************************************
  for( double freq = 4.0; freq <= 5.0; freq += 1.0/7.0)
  {
    cerr << "Frequency: " << freq << endl;

    const float amplitude = 1.0;

    // init buffers
    for (size_t i = 0; i < size; i++) {
      const double phase = (double)i * 2.0 * M_PI * freq / fftSize;
      const float real = amplitude * cos(phase);
      const float imag = amplitude * sin(phase);

      inout.get<fcomplex>()[i] = fcomplex(real, imag);
      f_inout[i][0] = real;
      f_inout[i][1] = imag;
    }

    // GPU: Forward FFT: compute and I/O
    stream.writeBuffer(d_inout, inout);
    fftFwdKernel.enqueue(blockId);
    stream.readBuffer(inout, d_inout, true);

    // FFTW: Forward FFT
    fftwf_execute(f_fftFwdPlan);

    // verify output -- we can't verify per-element because the leakage will
    // differ in pattern. So we collect the total signal leak outside our peak.
    double leak_gpu = 0.0;
    double leak_fftw = 0.0;

    // Check for our frequency response
    for (int i = 0; i < fftSize; i++) {
      if (i == (int)floor(freq) || i == (int)ceil(freq)) {
        /*
        if (!cmp_fcomplex(inout.get<fcomplex>()[i], fcomplex(amplitude * (float)fftSize, 0.0f))) {
          nrErrors += 1;
          cerr << "fwd: " << inout.get<fcomplex>()[i] << " at idx " << i << " should have been " << fcomplex((float)fftSize, 0.0f) << endl;
        }
        */

        cerr << "GPU  signal peak @ freq " << i << ": " << amp(inout.get<fcomplex>()[i]) << endl;
        cerr << "FFTW signal peak @ freq " << i << ": " << amp(f_inout[i]) << endl;
      } else {
        /*
        if (!cmp_fcomplex(inout.get<fcomplex>()[i], fcomplex(0.0f, 0.0f))) {
          if (++nrErrors < 100) { // limit spam
            cerr << "fwd: " << i << ':' << inout.get<fcomplex>()[i] << endl;
          }
        }
        */

        // accumulate leakage
        leak_gpu += amp(inout.get<fcomplex>()[i]);
        leak_fftw += amp(f_inout[i]);
      }
    }

    cerr << "GPU  signal leak: " << leak_gpu << endl;
    cerr << "FFTW signal leak: " << leak_fftw << endl;

    // Allow at most twice the leakage of FFTW, and accept any near-zero
    // leakage.
    double leak_error = abs(leak_gpu - leak_fftw) / leak_fftw;
    ASSERT(leak_gpu < 1.0e-4 || leak_error < 1.0);
  }

  if (nrErrors > 0)
  {
    cerr << "Error: " << nrErrors << " unexpected output values" << endl;
  }

  // Cleanup
  fftwf_destroy_plan(f_fftFwdPlan);
  fftwf_free(f_inout);
  fftw_cleanup_threads();

  // Done

  return nrErrors > 0;
}


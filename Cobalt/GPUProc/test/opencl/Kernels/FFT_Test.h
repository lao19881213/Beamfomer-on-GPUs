//# FFT_Test.h
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: FFT_Test.h 25313 2013-06-13 02:12:26Z amesfoort $

#ifndef GPUPROC_FFT_TEST_H
#define GPUPROC_FFT_TEST_H

#include <cstdlib>
#include <sys/time.h>
#include <cmath>
#include <cassert>
#include <fftw3.h>
#include <iostream>
#include <iomanip>

#include <UnitTest.h>
#include <GPUProc/Kernels/FFT_Kernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct FFT_Test : public UnitTest
    {
      FFT_Test(const Parset &ps)
        : UnitTest(ps, "FFT.cl")
      {
        bool testOk = true;
        unsigned nrErrors;

        const unsigned fftSize = 256;
        const unsigned nrFFTs = 1;
        MultiArraySharedBuffer<std::complex<float>, 1> inout(boost::extents[fftSize], queue, CL_MEM_READ_WRITE, CL_MEM_READ_WRITE);

        std::cout << "FFT Test" << std::endl;

        FFT_Kernel fftFwdKernel(context, fftSize, nrFFTs, true, inout);
        FFT_Kernel fftBwdKernel(context, fftSize, nrFFTs, false, inout);


        // First run two very basic tests, then a third test with many transforms with FFTW output as reference.
        // All tests run complex-to-complex float, in-place.

        // Test 1: Impulse at origin
        {
          std::cout << "FFT_Test 1" << std::endl;
          nrErrors = 0;

          inout[0] = 1.0f;

          inout.hostToDevice(CL_FALSE);
          fftFwdKernel.enqueue(queue, counter);
          inout.deviceToHost(CL_TRUE);

          // Check for constant function in transfer domain. All real values 1.0 (like fftw, scaled). All imag must be 0.0.
          for (unsigned i = 0; i < fftSize; i++) {
            if (inout[i] != 1.0f) {
              if (++nrErrors < 100) { // limit spam
                std::cerr << "fwd: " << i << ':' << inout[i] << std::endl;
              }
            }
          }

          // Backward
          fftBwdKernel.enqueue(queue, counter);
          inout.deviceToHost(CL_TRUE);

          // See if we got only our scaled impuls back.
          if (inout[0] != (float)fftSize) {
            nrErrors += 1;
            std::cerr << "bwd: " << inout[0] << " at idx 0 should have been " << (std::complex<float>)fftSize << std::endl;
          }
          for (unsigned i = 1; i < fftSize; i++) {
            if (inout[i] != 0.0f) {
              if (++nrErrors < 100) {
                std::cerr << "bwd: " << i << ':' << inout[i] << std::endl;
              }
            }
          }

          if (nrErrors == 0) {
            std::cout << "FFT_Test 1: test OK" << std::endl;
          } else {
            std::cerr << "FFT_Test 1: failed with " << nrErrors << " unexpected values" << std::endl;
            testOk = false;
          }
        }


        // Test 2: Shifted impulse
        {
          std::cout << "FFT_Test 2" << std::endl;
          nrErrors = 0;
          const float eps = 1.0e-4f;
          std::cout << "using epsilon = " << std::setprecision(9+1) << eps << std::endl;
          memset(inout.origin(), 0, inout.num_elements() * sizeof(std::complex<float>));

          inout[1] = 1.0f;

          inout.hostToDevice(CL_FALSE);
          fftFwdKernel.enqueue(queue, counter);
          inout.deviceToHost(CL_TRUE);

          // Check for (scaled) cosine real vals, and minus (scaled) sine imag vals.
          // (One could also roughly check that each complex val is of constant (scaled) magnitude (sin^2(x) + cos^2(x) = 1).)
          for (unsigned i = 0; i < fftSize; i++) {
            std::complex<float> ref = std::complex<float>((float) std::cos(2.0 * M_PI * i / fftSize),
                                                          (float)-std::sin(2.0 * M_PI * i / fftSize));
            if (!fpEquals(inout[i], ref, eps)) {
              if (++nrErrors < 100) {
                std::cerr << "fwd: " << inout[i] << " at idx " << i << " should have been " << inout[i] << std::endl;
              }
            }
          }

          // Backward
          fftBwdKernel.enqueue(queue, counter);
          inout.deviceToHost(CL_TRUE);

          // See if we got only our scaled, shifted impuls back.
          if (!fpEquals(inout[0], 0.0f, eps)) {
            nrErrors += 1;
            std::cerr << "bwd: " << inout[0] << " at idx 0 should have been (0.0, 0.0)" << std::endl;
          }
          if (!fpEquals(inout[1], (float)fftSize, eps)) {
            nrErrors += 1;
            std::cerr << "bwd: " << inout[1] << " at idx 1 should have been " << (std::complex<float>)fftSize << std::endl;
          }
          for (unsigned i = 2; i < fftSize; i++) {
            if (!fpEquals(inout[i], 0.0f, eps)) {
              if (++nrErrors < 100) {
                std::cerr << "bwd: " << i << ':' << inout[i] << std::endl;
              }
            }
          }

          if (nrErrors == 0) {
            std::cout << "FFT_Test 2: test OK" << std::endl;
          } else {
            std::cerr << "FFT_Test 2: failed with " << nrErrors << " unexpected values" << std::endl;
            testOk = false;
          }
        }


        // Test 3: Pseudo-random input ([0.0f, 2*pi]) on parset specified sizes. Compare output against FFTW double precision.
        {
          std::cout << "FFT_Test 3" << std::endl;
          nrErrors = 0;

          const float eps = 1.0e-3f;
          std::cout << "using epsilon = " << std::setprecision(9+1) << eps << std::endl;

          struct timeval tv = {0, 0};
          gettimeofday(&tv, NULL);
          const unsigned int seed = (unsigned int)tv.tv_sec;
          std::srand(seed);

          const unsigned fftSize = ps.nrChannelsPerSubband();
          const unsigned nrFFTs = ps.nrStations() * NR_POLARIZATIONS * ps.nrSamplesPerChannel();
          MultiArraySharedBuffer<std::complex<float>, 2> inout3(boost::extents[nrFFTs][fftSize], queue, CL_MEM_READ_WRITE, CL_MEM_READ_WRITE);

          FFT_Kernel fftFwdKernel3(context, fftSize, nrFFTs, true, inout3);
          FFT_Kernel fftBwdKernel3(context, fftSize, nrFFTs, false, inout3);

          fftw_plan fwdPlan;
          fftw_plan bwdPlan;
          fftw_complex* refInout3;

          bool fftwOk = fftwInit(&fwdPlan, &bwdPlan, &refInout3, fftSize, nrFFTs);
          assert(fftwOk);

          for (unsigned i = 0; i < inout3.num_elements(); i++) {
            double real = (std::rand() * 2.0 * M_PI / RAND_MAX);
            double imag = (std::rand() * 2.0 * M_PI / RAND_MAX);
            refInout3[i][0] = real;
            refInout3[i][1] = imag;
            inout3.origin()[i].real() = (float)real;
            inout3.origin()[i].imag() = (float)imag;
          }

          inout3.hostToDevice(CL_FALSE);
          fftFwdKernel3.enqueue(queue, counter);
          inout3.deviceToHost(CL_TRUE);

          fftw_execute(fwdPlan);

          float maxDiff = 0.0f;
          for (unsigned i = 0; i < nrFFTs; i++) {
            for (unsigned j = 0; j < fftSize; j++) {
              std::complex<float> fref;
              fref.real() = (float)refInout3[i * fftSize + j][0];
              fref.imag() = (float)refInout3[i * fftSize + j][1];
              if (!fpEquals(inout3[i][j], fref, eps)) {
                if (++nrErrors < 100) {
                  std::cerr << "fwd: " << inout3[i][j] << " at transform " << i << " pos " << j
                            << " should have been " << fref << std::endl;
                }
              }

              float diffReal = std::abs(inout3[i][j].real() - fref.real());
              if (diffReal > maxDiff)
                maxDiff = diffReal;
              float diffImag = std::abs(inout3[i][j].imag() - fref.imag());
              if (diffImag > maxDiff)
                maxDiff = diffImag;
            }
          }
          std::cout << "FFT_Test 3: Max abs error (fwd) compared to fftw complex double (w/ FFTW_ESTIMATE plans) is: " << maxDiff << std::endl;

          // Backward
          fftBwdKernel3.enqueue(queue, counter);
          inout3.deviceToHost(CL_TRUE);

          fftw_execute(bwdPlan);

          maxDiff = 0.0f;
          // Compare again vs fftw complex double. The original input has been overwritten.
          for (unsigned i = 0; i < nrFFTs; i++) {
            for (unsigned j = 0; j < fftSize; j++) {
              std::complex<float> fref;
              fref.real() = (float)refInout3[i * fftSize + j][0];
              fref.imag() = (float)refInout3[i * fftSize + j][1];
              if (!fpEquals(inout3[i][j], fref, eps)) {
                if (++nrErrors < 100) {
                  std::cerr << "bwd: " << inout3[i][j] << " at transform " << i << " pos " << j
                            << " should have been " << fref << std::endl;
                }
              }

              float diffReal = std::abs(inout3[i][j].real() - fref.real());
              if (diffReal > maxDiff)
                maxDiff = diffReal;
              float diffImag = std::abs(inout3[i][j].imag() - fref.imag());
              if (diffImag > maxDiff)
                maxDiff = diffImag;
            }
          }
          std::cout << "FFT_Test 3: Max abs error (fwd+bwd) compared to fftw complex double (w/ FFTW_ESTIMATE plans) is: " << maxDiff << std::endl;

          fftwDeinit(fwdPlan, bwdPlan, refInout3);

          if (nrErrors == 0) {
            std::cout << "FFT_Test 3: test OK" << std::endl;
          } else {
            std::cerr << "FFT_Test 3: failed with " << nrErrors << " unexpected values" << std::endl;
            testOk = false;
          }
        }


        check(testOk, true);
      }


      bool fftwInit(fftw_plan* fwdPlan, fftw_plan* bwdPlan, fftw_complex** inout, int fftSize, int nrFFTs)
      {
        if (fftw_init_threads() == 0) {
          std::cerr << "failed to init fftw threads" << std::endl;
          return false;
        }

        *inout = (fftw_complex*)fftw_malloc(fftSize * nrFFTs * sizeof(fftw_complex));
        if (*inout == NULL) {
          std::cerr << "failed to fftw malloc buffer" << std::endl;
          fftw_cleanup_threads();
          return false;
        }

        fftw_plan_with_nthreads(4); // use up to 4 threads (don't care about test performance, but be impatient anyway...)

        // Use FFTW_ESTIMATE: we need reference output, so don't care about runtime speed.
        *fwdPlan = fftw_plan_many_dft(1, &fftSize, nrFFTs,         // int rank, const int *n (=dims), int howmany,
                                      *inout, NULL, 1, fftSize,    // fftw_complex *in, const int *inembed, int istride, int idist,
                                      *inout, NULL, 1, fftSize,    // fftw_complex *out, const int *onembed, int ostride, int odist,
                                      FFTW_FORWARD, FFTW_ESTIMATE); // int sign, unsigned flags
        if (*fwdPlan == NULL) {
          std::cerr << "failed to create fftw fwd plan" << std::endl;
          fftw_free(*inout);
          fftw_cleanup_threads();
          return false;
        }
        *bwdPlan = fftw_plan_many_dft(1, &fftSize, nrFFTs,         // int rank, const int *n (=dims), int howmany,
                                      *inout, NULL, 1, fftSize,    // fftw_complex *in, const int *inembed, int istride, int idist,
                                      *inout, NULL, 1, fftSize,    // fftw_complex *out, const int *onembed, int ostride, int odist,
                                      FFTW_BACKWARD, FFTW_ESTIMATE); // int sign, unsigned flags
        if (*bwdPlan == NULL) {
          std::cerr << "failed to create fftw bwd plan" << std::endl;
          fftw_destroy_plan(*fwdPlan);
          fftw_free(*inout);
          fftw_cleanup_threads();
          return false;
        }

        return true;
      }

      void fftwDeinit(fftw_plan fwdPlan, fftw_plan bwdPlan, fftw_complex* inout)
      {
        fftw_destroy_plan(bwdPlan);
        fftw_destroy_plan(fwdPlan);
        fftw_free(inout);
        fftw_cleanup_threads();
      }
    };
  }
}

#endif


//# AMD_FFT_Test.h
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
//# $Id: AMD_FFT_Test.h 24388 2013-03-26 11:14:29Z amesfoort $

#ifndef GPUPROC_AMD_FFT_TEST_H
#define GPUPROC_AMD_FFT_TEST_H

#include <UnitTest.h>
#include <clAmdFft.h>

namespace LOFAR
{
  namespace Cobalt
  {
    struct AMD_FFT_Test : public UnitTest
    {
#if 0
      AMD_FFT_Test(const Parset &ps)
        : UnitTest(ps, "fft2.cl")
      {
        MultiArraySharedBuffer<std::complex<float>, 1> in(boost::extents[8], queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
        MultiArraySharedBuffer<std::complex<float>, 1> out(boost::extents[8], queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);

        std::cout << "AMD FFT Test" << std::endl;

        for (unsigned i = 0; i < 8; i++)
          in[i] = std::complex<float>(2 * i + 1, 2 * i + 2);

        clAmdFftSetupData setupData;
        cl::detail::errHandler(clAmdFftInitSetupData(&setupData), "clAmdFftInitSetupData");
        setupData.debugFlags = CLFFT_DUMP_PROGRAMS;
        cl::detail::errHandler(clAmdFftSetup(&setupData), "clAmdFftSetup");

        clAmdFftPlanHandle plan;
        size_t dim[1] = { 8 };

        cl::detail::errHandler(clAmdFftCreateDefaultPlan(&plan, context(), CLFFT_1D, dim), "clAmdFftCreateDefaultPlan");
        cl::detail::errHandler(clAmdFftSetResultLocation(plan, CLFFT_OUTOFPLACE), "clAmdFftSetResultLocation");
        cl::detail::errHandler(clAmdFftSetPlanBatchSize(plan, 1), "clAmdFftSetPlanBatchSize");
        cl::detail::errHandler(clAmdFftBakePlan(plan, 1, &queue(), 0, 0), "clAmdFftBakePlan");

        in.hostToDevice(CL_FALSE);
        cl_mem ins[1] = { ((cl::Buffer) in)() };
        cl_mem outs[1] = { ((cl::Buffer) out)() };
#if 1
        cl::detail::errHandler(clAmdFftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue(), 0, 0, 0, ins, outs, 0), "clAmdFftEnqueueTransform");
#else
        cl::Kernel kernel(program, "fft_fwd");
        kernel.setArg(0, (cl::Buffer) in);
        kernel.setArg(1, (cl::Buffer) out);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(64, 1, 1), cl::NDRange(64, 1, 1));
#endif
        out.deviceToHost(CL_TRUE);

        for (unsigned i = 0; i < 8; i++)
          std::cout << out[i] << std::endl;

        cl::detail::errHandler(clAmdFftDestroyPlan(&plan), "clAmdFftDestroyPlan");
        cl::detail::errHandler(clAmdFftTeardown(), "clAmdFftTeardown");
      }

#endif

    };
  }
}

#endif


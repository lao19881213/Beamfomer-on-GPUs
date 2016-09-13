//# tBandPassCorrectionKernel2.cc: test BandPassCorrectionKernel class
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
//# $Id: tBandPassCorrectionKernel2.cc 27265 2013-11-06 13:28:18Z klijn $

#include <lofar_config.h>
#include <GPUProc/Kernels/BandPassCorrectionKernel.h>
#include <CoInterface/Parset.h>
#include <Common/lofar_iostream.h>
#include <UnitTest++.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;


struct TestFixture
{
  TestFixture() : 
    ps("tBandPassCorrectionKernel2.in_parset"), 
    params(ps),
    factory()
    {
      // Default parameters parsed from parset are not correct
      params.nrChannels1 = 64;
      params.nrChannels2 = 64;
      params.nrSamplesPerChannel = 
          ps.nrSamplesPerSubband() / (params.nrChannels1 * params.nrChannels2);

     factory = new KernelFactory<BandPassCorrectionKernel>(params);
    }

  ~TestFixture() {}

  Parset ps;
  BandPassCorrectionKernel::Parameters params;
  KernelFactory<BandPassCorrectionKernel> *factory;
};

TEST_FIXTURE(TestFixture, InputData)
{
  CHECK_EQUAL(size_t(1 * 2 * 64 *48 * 64 * 8),  // 1 station, 2 pol, 64 channels, 48 samples/channel, 64 2nd channel , n bytes in complex float
              factory->bufferSize(
                BandPassCorrectionKernel::INPUT_DATA));
}

TEST_FIXTURE(TestFixture, OutputData)
{
  CHECK_EQUAL(size_t(3145728),
              factory->bufferSize(
                BandPassCorrectionKernel::OUTPUT_DATA));
}


TEST_FIXTURE(TestFixture, BandPassCorrectionWeights)
{
  CHECK_EQUAL(size_t(64 * 64 * 4),
              factory->bufferSize(
                BandPassCorrectionKernel::BAND_PASS_CORRECTION_WEIGHTS));
}

TEST_FIXTURE(TestFixture, MustThrow)
{
  CHECK_THROW(factory->bufferSize(
                BandPassCorrectionKernel::BufferType(5)),
              GPUProcException);
}

int main()
{
  INIT_LOGGER("tBandPassCorrectionKernel2");
  try {
    gpu::Platform pf;
  } catch (gpu::GPUException&) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 3;
  }
  return UnitTest::RunAllTests() > 0;
}

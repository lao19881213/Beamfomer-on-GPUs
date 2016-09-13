//# tFIR_FilterKernel.cc: test FIR_FilterKernel class
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
//# $Id: tFIR_FilterKernel.cc 26758 2013-09-30 11:47:21Z loose $

#include <lofar_config.h>

#include <GPUProc/Kernels/FIR_FilterKernel.h>
#include <CoInterface/Parset.h>
#include <Common/lofar_complex.h>

#include <UnitTest++.h>
#include <memory>
#include <GPUProc/PerformanceCounter.h>
#include <CoInterface/BlockID.h>

using namespace LOFAR::Cobalt;
using namespace LOFAR;
using namespace std;

TEST(FIR_FilterKernel)
{
  Parset ps;
  ps.add("Observation.nrBitsPerSample", "8");
  ps.add("Observation.VirtualInstrument.stationList", "[RS000]");
  ps.add("OLAP.CNProc.integrationSteps", "128");
  ps.add("Observation.channelsPerSubband", "64");
  ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
  ps.updateSettings();

  KernelFactory<FIR_FilterKernel> factory(ps);

  gpu::Device device(gpu::Platform().devices()[0]);
  gpu::Context context(device);
  gpu::Stream stream(context);

  gpu::DeviceMemory
    dInput(context, factory.bufferSize(FIR_FilterKernel::INPUT_DATA)),
    dOutput(context, factory.bufferSize(FIR_FilterKernel::OUTPUT_DATA)),
    dCoeff(context, factory.bufferSize(FIR_FilterKernel::FILTER_WEIGHTS)),
    dHistory(context, factory.bufferSize(FIR_FilterKernel::HISTORY_DATA));

  gpu::HostMemory
    hInput(context, dInput.size()),
    hOutput(context, dOutput.size()),
    hCoeff(context, dCoeff.size()),
    hHistory(context, dHistory.size());

  cout << "dInput.size() = " << dInput.size() << endl;
  cout << "dOutput.size() = " << dOutput.size() << endl;
  cout << "dCoeff.size() = " << dCoeff.size() << endl;
  cout << "dHistory.size() = " << dHistory.size() << endl;

  // hInput.get<i8complex>()[2176] = i8complex(1,0);

  i8complex* ibuf = hInput.get<i8complex>();
  for(size_t i = 1922; i < 1923; ++i) {
    ibuf[i] = i8complex(1,0);
  }

  stream.writeBuffer(dInput, hInput);

  // initialize history data
  dHistory.set(0);

  FIR_FilterKernel::Buffers buffers(dInput, dOutput, dCoeff, dHistory);
  auto_ptr<FIR_FilterKernel> kernel(factory.create(stream, buffers));
  PerformanceCounter counter(context);
  BlockID blockId;
  kernel->enqueue(blockId, counter, 0);

  stream.readBuffer(hOutput, dOutput);
  stream.readBuffer(hCoeff, dCoeff);

  /*  Comment out printing of this information: it disrupts the logfile and add no information.
  float* buf = hOutput.get<float>();
  for(size_t i = 0; i < hOutput.size() / sizeof(float); ++i) {
    cout << "out[" << i << "] = " << buf[i] << endl;
  }

  buf = hCoeff.get<float>();
  for(size_t i = 0; i < hCoeff.size() / sizeof(float); ++i) {
    cout << "coeff[" << i << "] = " << buf[i] << endl;
  }
  */
}

TEST(HistoryFlags)
{
  /*
   * Set up a Kernel
   */

  Parset ps;
  ps.add("Observation.nrBitsPerSample", "8");
  ps.add("Observation.VirtualInstrument.stationList", "[RS000]");
  ps.add("OLAP.CNProc.integrationSteps", "128");
  ps.add("Observation.channelsPerSubband", "64");
  ps.add("Observation.DataProducts.Output_Correlated.enabled", "true");
  ps.updateSettings();

  FIR_FilterKernel::Parameters params(ps);

  KernelFactory<FIR_FilterKernel> factory(params);

  gpu::Device device(gpu::Platform().devices()[0]);
  gpu::Context context(device);
  gpu::Stream stream(context);

  gpu::DeviceMemory
    dInput(context, factory.bufferSize(FIR_FilterKernel::INPUT_DATA)),
    dOutput(context, factory.bufferSize(FIR_FilterKernel::OUTPUT_DATA)),
    dCoeff(context, factory.bufferSize(FIR_FilterKernel::FILTER_WEIGHTS)),
    dHistory(context, factory.bufferSize(FIR_FilterKernel::HISTORY_DATA));

  FIR_FilterKernel::Buffers buffers(dInput, dOutput, dCoeff, dHistory);
  auto_ptr<FIR_FilterKernel> kernel(factory.create(stream, buffers));

  /*
   * Test propagation of history flags. Each block tests for the flags of
   * the history samples of the previous block, so the order of these tests
   * matters.
   */

  MultiDimArray<SparseSet<unsigned>, 1> inputFlags(boost::extents[1]);

  /*
   * Block 0: only last sample is flagged
   */

  // Flag only the last sample
  inputFlags[0].reset();
  inputFlags[0].include(ps.nrSamplesPerSubband() - 1);

  // insert and update history flags
  kernel->prefixHistoryFlags(inputFlags, 0);

  // the first set of history flags are all flagged, and so is our last sample
  CHECK_EQUAL(params.nrHistorySamples() + 1, inputFlags[0].count());

  /*
   * Block 1: no samples are flagged
   */

  // next block
  inputFlags[0].reset();
  kernel->prefixHistoryFlags(inputFlags, 0);

  // the second set of history flags should have one sample flagged (the last
  // sample of the previous block)
  CHECK_EQUAL(1U, inputFlags[0].count());

  /*
   * Block 2: all samples are flagged
   */

  // next block
  inputFlags[0].reset();
  inputFlags[0].include(0, ps.nrSamplesPerSubband());
  kernel->prefixHistoryFlags(inputFlags, 0);

  // the number of flagged samples should have remained unchanged (the last
  // block had no flags)
  CHECK_EQUAL(ps.nrSamplesPerSubband(), inputFlags[0].count());

  /*
   * Block 3: no samples are flagged
   */

  // next block
  inputFlags[0].reset();
  kernel->prefixHistoryFlags(inputFlags, 0);

  // only the history samples should be flagged
  CHECK_EQUAL(params.nrHistorySamples(), inputFlags[0].count());
  CHECK_EQUAL(params.nrHistorySamples(), inputFlags[0].subset(0, params.nrHistorySamples()).count());
}

int main()
{
  INIT_LOGGER("tFIR_FilterKernel");
  try {
    gpu::Platform pf;
    return UnitTest::RunAllTests() > 0;
  } catch (gpu::GPUException& e) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 3;
  }

}


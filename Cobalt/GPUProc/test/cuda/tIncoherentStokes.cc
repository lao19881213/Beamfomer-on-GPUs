//# tIncoherentStokes.cc: test incoherent Stokes kernel
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
//# $Id: tIncoherentStokes.cc 27228 2013-11-04 12:40:38Z loose $

#include <lofar_config.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/Kernels/IncoherentStokesKernel.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <CoInterface/BlockID.h>
#include <CoInterface/Parset.h>
#include <Common/LofarLogger.h>

#include <UnitTest++.h>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace boost;
using namespace LOFAR::Cobalt;

typedef complex<float> fcomplex;

// Sine and cosine tables for angles that are multiples of 30 degrees in the
// range [0 .. 360> degrees.
float sine[]   = { 0,  0.5,  0.86602540478,  1,  0.86602540478,  0.5,
                   0, -0.5, -0.86602540478, -1, -0.86602540478, -0.5 };
float cosine[] = { 1,  0.86602540478,  0.5,  0, -0.5, -0.86602540478,
                  -1, -0.86602540478, -0.5,  0,  0.5,  0.86602540478 };


// Fixture for testing correct translation of parset values
struct ParsetFixture
{
  static const size_t 
    timeIntegrationFactor = sizeof(sine) / sizeof(float),
    nrChannels = 37,
    nrOutputSamples = 29,
    nrInputSamples = nrOutputSamples * timeIntegrationFactor, 
    blockSize = timeIntegrationFactor * nrChannels * nrInputSamples,
    nrStations = 43;

  Parset parset;

  ParsetFixture() {
    parset.add("Observation.DataProducts.Output_Beamformed.enabled", 
               "true");
    parset.add("OLAP.CNProc_IncoherentStokes.timeIntegrationFactor", 
               lexical_cast<string>(timeIntegrationFactor));
    parset.add("OLAP.CNProc_IncoherentStokes.channelsPerSubband",
               lexical_cast<string>(nrChannels));
    parset.add("OLAP.CNProc_IncoherentStokes.which",
               "IQUV");
    parset.add("Observation.VirtualInstrument.stationList",
               str(format("[%d*RS000]") % nrStations));
    parset.add("Cobalt.blockSize", 
               lexical_cast<string>(blockSize)); 
    parset.updateSettings();
  }
};

const size_t
  ParsetFixture::timeIntegrationFactor,
  ParsetFixture::nrChannels,
  ParsetFixture::nrOutputSamples,
  ParsetFixture::nrInputSamples,
  ParsetFixture::blockSize,
  ParsetFixture::nrStations;


// Test correctness of reported buffer sizes
TEST_FIXTURE(ParsetFixture, BufferSizes)
{
  const ObservationSettings::BeamFormer::StokesSettings &settings = 
    parset.settings.beamFormer.incoherentSettings;
  CHECK_EQUAL(timeIntegrationFactor, settings.timeIntegrationFactor);
  CHECK_EQUAL(nrChannels, settings.nrChannels);
  CHECK_EQUAL(4U, settings.nrStokes);
  CHECK_EQUAL(nrStations, parset.nrStations());
  CHECK_EQUAL(nrInputSamples, settings.nrSamples(blockSize));
}


// Test if we can succesfully create a KernelFactory
TEST_FIXTURE(ParsetFixture, KernelFactory)
{
  KernelFactory<IncoherentStokesKernel> kf(parset);
}


// Fixture for testing the IncoherentStokes kernel itself.
struct KernelFixture : ParsetFixture
{
  gpu::Device device;
  gpu::Context context;
  gpu::Stream stream;
  size_t nrStokes;
  KernelFactory<IncoherentStokesKernel> factory;
  MultiDimArrayHostBuffer<fcomplex, 4> hInput;
  MultiDimArrayHostBuffer<float, 3> hOutput;
  MultiDimArrayHostBuffer<float, 3> hRefOutput;
  IncoherentStokesKernel::Buffers buffers;
  scoped_ptr<IncoherentStokesKernel> kernel;

  KernelFixture() :
    device(gpu::Platform().devices()[0]),
    context(device),
    stream(context),
    nrStokes(parset.settings.beamFormer.incoherentSettings.nrStokes),
    factory(parset),
    hInput(
      boost::extents[nrStations][NR_POLARIZATIONS][nrInputSamples][nrChannels],
      context),
    hOutput(
      boost::extents[nrStokes][nrOutputSamples][nrChannels],
      context),
    hRefOutput(
      boost::extents[nrStokes][nrOutputSamples][nrChannels],
      context),
    buffers(
      gpu::DeviceMemory(
        context, factory.bufferSize(IncoherentStokesKernel::INPUT_DATA)),
      gpu::DeviceMemory(
        context, factory.bufferSize(IncoherentStokesKernel::OUTPUT_DATA))),
    kernel(factory.create(stream, buffers))
  {
    initializeHostBuffers();
  }

  // Initialize all the elements of the input host buffer to zero, and all
  // elements of the output host buffer to NaN.
  void initializeHostBuffers()
  {
    cout << "\nInitializing host buffers..."
         // << "\n  timeIntegrationFactor = " << setw(7) << timeIntegrationFactor
         // << "\n  nrChannels            = " << setw(7) << nrChannels
         // << "\n  nrInputSamples        = " << setw(7) << nrInputSamples
         // << "\n  nrOutputSamples       = " << setw(7) << nrOutputSamples
         // << "\n  nrStations            = " << setw(7) << nrStations
         // << "\n  blockSize             = " << setw(7) << blockSize
         << "\n  buffers.input.size()  = " << setw(7) << buffers.input.size()
         << "\n  buffers.output.size() = " << setw(7) << buffers.output.size()
         << endl;
    CHECK_EQUAL(buffers.input.size(), hInput.size());
    CHECK_EQUAL(buffers.output.size(), hOutput.size());
    fill(hInput.data(), hInput.data() + hInput.num_elements(), 0);
    fill(hOutput.data(), hOutput.data() + hOutput.num_elements(), 0.0f / 0.0f);
    fill(hRefOutput.data(), hRefOutput.data() + hRefOutput.num_elements(), 0);
  }

  void runKernel()
  {
    // Dummy BlockID
    BlockID blockId;
    // Copy input data from host- to device buffer synchronously
    stream.writeBuffer(buffers.input, hInput, true);
    // Launch the kernel
    kernel->enqueue(blockId);
    // Copy output data from device- to host buffer synchronously
    stream.readBuffer(hOutput, buffers.output, true);
  }

  // void printNonZeroOutput() const
  // {
  //   for (size_t stokes = 0; stokes < nrStokes; stokes++) {
  //     for (size_t time = 0; time < nrOutputSamples; time++) {
  //       for (size_t chan = 0; chan < nrChannels; chan++) {
  //         if (hOutput[stokes][time][chan] != 0) {
  //           cout << "hOutput[" << stokes << "][" << time << "][" << chan 
  //                << "] = " << hOutput[stokes][time][chan] << endl;
  //         }
  //       }
  //     }
  //   }
  // }

};


// An input of all zeros should result in an output of all zeros.
TEST_FIXTURE(KernelFixture, ZeroTest)
{
  // Host buffers are properly initialized for this test. Just run the kernel.
  runKernel();
  CHECK_ARRAY_EQUAL(hRefOutput.data(), hOutput.data(), hOutput.num_elements());
}


// Input differently polarized signals into different channels and check the
// output for correct Stoke I,Q,U,V values.
TEST_FIXTURE(KernelFixture, StokesTest)
{
  // Initialize input host memory and reference output.
  const size_t N = sizeof(sine) / sizeof(float);
  for(size_t n = 0; n < nrInputSamples; n++)
  {
    // Station 1, channel 3: linear polarization in X  ==>  I, +Q
    hInput[1][0][n][3] = fcomplex(cosine[n%N], sine[n%N]);
    if (n < nrOutputSamples) {
      hRefOutput[0][n][3] = float(timeIntegrationFactor);  // Stokes I
      hRefOutput[1][n][3] = float(timeIntegrationFactor);  // Stokes Q
    }
    // Station 4, channel 7: linear polarization in Y  ==>  I, -Q
    hInput[4][1][n][7] = fcomplex(cosine[n%N], sine[n%N]);
    if (n < nrOutputSamples) {
      hRefOutput[0][n][7] = float(timeIntegrationFactor);  // Stokes I
      hRefOutput[1][n][7] = -float(timeIntegrationFactor); // Stokes Q
    }
    // Station 6, channel 8: linear polarization in X and Y  ==>  I, +U
    hInput[6][0][n][8] = fcomplex(cosine[n%N], sine[n%N]);
    hInput[6][1][n][8] = fcomplex(cosine[n%N], sine[n%N]);
    if (n < nrOutputSamples) {
      hRefOutput[0][n][8] = float(2 * timeIntegrationFactor);  // Stokes I
      hRefOutput[2][n][8] = float(2 * timeIntegrationFactor);  // Stokes U
    }
    // Station 7, channel 11: linear polarization in X and -Y  ==>  I, -U
    hInput[7][0][n][11] = fcomplex(cosine[n%N], sine[n%N]);
    hInput[7][1][n][11] = -fcomplex(cosine[n%N], sine[n%N]);
    if (n < nrOutputSamples) {
      hRefOutput[0][n][11] = float(2 * timeIntegrationFactor);  // Stokes I
      hRefOutput[2][n][11] = -float(2 * timeIntegrationFactor); // Stokes U
    }
    // Station 10, channel 13: right circular polarization  ==>  I, +V
    hInput[10][0][n][13] = fcomplex(cosine[n%N], sine[n%N]);
    hInput[10][1][n][13] = fcomplex(sine[n%N], -cosine[n%N]);
    if (n < nrOutputSamples) {
      hRefOutput[0][n][13] = float(2 * timeIntegrationFactor);  // Stokes I
      hRefOutput[3][n][13] = float(2 * timeIntegrationFactor);  // Stokes V
    }
    // Station 12, channel 17: left circular polarization  ==>  I, -V
    hInput[12][0][n][17] = fcomplex(cosine[n%N], sine[n%N]);
    hInput[12][1][n][17] = fcomplex(-sine[n%N], cosine[n%N]);
    if (n < nrOutputSamples) {
      hRefOutput[0][n][17] = float(2 * timeIntegrationFactor);  // Stokes I
      hRefOutput[3][n][17] = -float(2 * timeIntegrationFactor); // Stokes V
    }
  }
  runKernel();
  CHECK_ARRAY_EQUAL(hRefOutput.data(), hOutput.data(), hOutput.num_elements());
}


int main()
{
  INIT_LOGGER("tIncoherentStokes");
  try {
    gpu::Platform pf;
    return UnitTest::RunAllTests() > 0;
  } catch (gpu::GPUException& e) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 3;
  }
}

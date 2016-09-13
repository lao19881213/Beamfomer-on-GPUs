//# tCorrelatorWorKQueueProcessSb.cc: test CorrelatorSubbandProc::processSubband()
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
//# $Id: tCorrelatorSubbandProcProcessSb.cc 26753 2013-09-30 09:05:45Z mol $

#include <lofar_config.h>

#include <complex>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/SubbandProcs/CorrelatorSubbandProc.h>

using namespace std;
using namespace LOFAR::Cobalt;

int main() {
  INIT_LOGGER("tCorrelatorSubbandProcProcessSb");

  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    return 3;
  }

  gpu::Device device(0);
  vector<gpu::Device> devices(1, device);
  gpu::Context ctx(device);

  Parset ps("tCorrelatorSubbandProcProcessSb.parset");

  // Create very simple kernel programs, with predictable output. Skip as much as possible.
  // Nr of channels/sb from the parset is 1, so the PPF will not even run.
  // Parset also has turned of delay compensation and bandpass correction
  // (but that kernel will run to convert int to float and to transform the data order).

  CorrelatorFactories factories(ps);

  cout << "FIR_FilterKernel::INPUT_DATA : "
       << factories.firFilter.bufferSize(FIR_FilterKernel::INPUT_DATA) << endl;
  cout << "FIR_FilterKernel::OUTPUT_DATA : "
       << factories.firFilter.bufferSize(FIR_FilterKernel::OUTPUT_DATA) << endl;
  cout << "FIR_FilterKernel::FILTER_WEIGHTS : "
       << factories.firFilter.bufferSize(FIR_FilterKernel::FILTER_WEIGHTS) << endl;
  cout << "FIR_FilterKernel::HISTORY_DATA : "
       << factories.firFilter.bufferSize(FIR_FilterKernel::HISTORY_DATA) << endl;
  CorrelatorSubbandProc cwq(ps, ctx, factories);

  SubbandProcInputData in(ps.nrBeams(), ps.nrStations(), ps.settings.nrPolarisations,
                          ps.settings.beamFormer.maxNrTABsPerSAP(),
                           ps.nrSamplesPerSubband(), ps.nrBytesPerComplexSample(), ctx);
  cout << "#st=" << ps.nrStations() << " #sampl/sb=" << ps.nrSamplesPerSubband() <<
          " #bytes/complexSampl=" << ps.nrBytesPerComplexSample() <<
          " Total bytes=" << in.inputSamples.size() << endl;

  // Initialize synthetic input to all (1, 1).
  for (size_t st = 0; st < ps.nrStations(); st++)
    for (size_t i = 0; i < ps.nrSamplesPerSubband(); i++)
      for (size_t pol = 0; pol < NR_POLARIZATIONS; pol++)
      {
        if (ps.nrBytesPerComplexSample() == 4) { // 16 bit mode
          *(int16_t *)&in.inputSamples[st][i][pol][0] = 1; // real
          *(int16_t *)&in.inputSamples[st][i][pol][2] = 1; // imag starts at byte idx 2
        } else if (ps.nrBytesPerComplexSample() == 2) { // 8 bit mod
          in.inputSamples[st][i][pol][0] = 1; // real
          in.inputSamples[st][i][pol][1] = 1; // imag
        } else {
          cerr << "Error: number of bits per sample must be 8, or 16" << endl;
          exit(1);
        }
      }

  // Initialize subbands partitioning administration (struct BlockID). We only do the 1st block of whatever.
  in.blockID.block = 0;            // Block number: 0 .. inf
  in.blockID.globalSubbandIdx = 0; // Subband index in the observation: [0, ps.nrSubbands())
  in.blockID.localSubbandIdx = 0;  // Subband index for this pipeline: [0, subbandIndices.size())
  in.blockID.subbandProcSubbandIdx = 0; // Subband index for this subbandProc: [0, nrSubbandsPerSubbandProc)
  in.blockID.subbandProcSubbandIdx = 0; // Subband index for this SubbandProc

  // Initialize delays. We skip delay compensation, but init anyway,
  // so we won't copy uninitialized data to the device.
  for (size_t i = 0; i < in.delaysAtBegin.size(); i++)
    in.delaysAtBegin.get<float>()[i] = 0.0f;
  for (size_t i = 0; i < in.delaysAfterEnd.size(); i++)
    in.delaysAfterEnd.get<float>()[i] = 0.0f;
  for (size_t i = 0; i < in.phaseOffsets.size(); i++)
    in.phaseOffsets.get<float>()[i] = 0.0f;

  CorrelatedDataHostBuffer out(ps.nrStations(), ps.nrChannelsPerSubband(),
                               ps.integrationSteps(), ctx);

  // Don't bother initializing out.blockID; processSubband() doesn't need it.

  cout << "processSubband()" << endl;
  cwq.processSubband(in, out);
  cout << "processSubband() done" << endl;

  cout << "Output: " << endl;
  unsigned nbaselines = ps.nrStations() * (ps.nrStations() + 1) / 2; // nbaselines includes auto-correlation pairs here
  cout << "nbl(w/ autocorr)=" << nbaselines << " #bytes/complexSample=" << ps.nrBytesPerComplexSample() <<
          " #chnl/sb=" << ps.nrChannelsPerSubband() << " #pol=" << NR_POLARIZATIONS <<
          " (all combos, hence x2) Total bytes=" << out.size() << endl;

  // Output verification
  // The int2float conversion scales its output to the same amplitude as in 16 bit mode.
  // For 8 bit mode, that is a factor 256.
  // Since we inserted all (1, 1) vals, for 8 bit mode this means that the correlator
  // outputs 16*16. It then sums over nrSamplesPerSb values.
  unsigned scale = 1*1;
  if (ps.nrBitsPerSample() == 8)
    scale = 16*16;
  bool unexpValueFound = false;
  for (size_t b = 0; b < nbaselines; b++)
    for (size_t c = 0; c < ps.nrChannelsPerSubband(); c++)
      // combinations of polarizations; what the heck, call it pol0 and pol1, but 2, in total 4.
      for (size_t pol0 = 0; pol0 < NR_POLARIZATIONS; pol0++)
        for (size_t pol1 = 0; pol1 < NR_POLARIZATIONS; pol1++)
        {
          complex<float> v = out[b][c][pol0][pol1];
          if (v.real() != static_cast<float>(scale * 2*ps.nrSamplesPerSubband()) ||
              v.imag() != 0.0f)
          {
            unexpValueFound = true;
            cout << '*'; // indicate error in output
          }
          cout << out[b][c][pol0][pol1] << " ";
        }
  cout << endl;

  if (unexpValueFound)
  {
    cerr << "Error: Found unexpected output value(s)" << endl;
    return 1;
  }

  // postprocessSubband() is about flagging and that has already been tested
  // in the other CorrelatorSubbandProc test.

  return 0;
}


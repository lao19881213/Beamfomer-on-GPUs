//# tBeamFormerSubbandProcProcessSb: test of Beamformer subband processor.
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
//# $Id: tCorrelatorSubbandProcProcessSb.cc 26496 2013-09-11 12:58:23Z mol $

#include <lofar_config.h>

#include <complex>
#include <cmath>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/SubbandProcs/BeamFormerSubbandProc.h>
#include <GPUProc/SubbandProcs/BeamFormerFactories.h>

#include "../fpequals.h"

using namespace std;
using namespace LOFAR::Cobalt;
using namespace LOFAR::TYPES;

const unsigned nrChannel1 = 64;
const unsigned nrChannel2 = 64;

template<typename T> T inputSignal(size_t t)
{
  size_t nrBits = sizeof(T) / 2 * 8;
  double freq = 1.0 / 4.0; // in samples
  // double freq = (2 * 64.0 + 17.0) / 4096.0; // in samples
  double amp = (1 << (nrBits - 1)) - 1;

  double angle = (double)t * 2.0 * M_PI * freq;

  double s = ::sin(angle);
  double c = ::cos(angle);

  return T(::round(amp * c), ::round(amp * s));
}

int main() {
  INIT_LOGGER("tBeamFormerSubbandProcProcessSb");

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

  Parset ps("tBeamFormerSubbandProcProcessSb.parset");

  // Input array sizes
  const size_t nrBeams = ps.nrBeams();
  const size_t nrStations = ps.nrStations();
  const size_t nrPolarisations = ps.settings.nrPolarisations;
  const size_t maxNrTABsPerSAP = ps.settings.beamFormer.maxNrTABsPerSAP();
  const size_t nrSamplesPerSubband = ps.nrSamplesPerSubband();
  const size_t nrBitsPerSample = ps.settings.nrBitsPerSample;
  const size_t nrBytesPerComplexSample = ps.nrBytesPerComplexSample();

  // We only support 8-bit or 16-bit input samples
  ASSERT(nrBitsPerSample == 8 || nrBitsPerSample == 16);

  // TODO: Amplitude is now calculated at two places. Dangerous!
  const size_t amplitude = (1 << (nrBitsPerSample - 1)) - 1;
  const size_t scaleFactor = nrBitsPerSample == 16 ? 1 : 16;

  LOG_INFO_STR(
    "Input info:" <<
    "\n  nrBeams = " << nrBeams <<
    "\n  nrStations = " << nrStations <<
    "\n  nrPolarisations = " << nrPolarisations <<
    "\n  maxNrTABsPerSAP = " << maxNrTABsPerSAP <<
    "\n  nrSamplesPerSubband = " << nrSamplesPerSubband <<
    "\n  nrBitsPerSample = " << nrBitsPerSample <<
    "\n  nrBytesPerComplexSample = " << nrBytesPerComplexSample);

  // Output array sizes
  const size_t nrStokes = ps.settings.beamFormer.incoherentSettings.nrStokes;
  const size_t nrChannels = 
    ps.settings.beamFormer.incoherentSettings.nrChannels;
  const size_t nrSamples = 
    ps.settings.beamFormer.incoherentSettings.nrSamples(
      ps.settings.nrSamplesPerSubband());

  LOG_INFO_STR(
    "Output info:" <<
    "\n  nrStokes = " << nrStokes <<
    "\n  nrChannels = " << nrChannels <<
    "\n  nrSamples = " << nrSamples <<
    "\n  scaleFactor = " << scaleFactor);

  // Create very simple kernel programs, with predictable output. Skip as much
  // as possible. Nr of channels/sb from the parset is 1, so the PPF will not
  // even run. Parset also has turned of delay compensation and bandpass
  // correction (but that kernel will run to convert int to float and to
  // transform the data order).

  BeamFormerFactories factories(ps);
  BeamFormerSubbandProc bwq(ps, ctx, factories);

  SubbandProcInputData in(
    nrBeams, nrStations, nrPolarisations, maxNrTABsPerSAP, 
    nrSamplesPerSubband, nrBytesPerComplexSample, ctx);

  // Initialize synthetic input to input signal
  for (size_t st = 0; st < nrStations; st++)
    for (size_t i = 0; i < nrSamplesPerSubband; i++)
      for (size_t pol = 0; pol < nrPolarisations; pol++)
      {
        switch(nrBitsPerSample) {
        case 8:
          reinterpret_cast<i8complex&>(in.inputSamples[st][i][pol][0]) =
            inputSignal<i8complex>(i);
          break;
        case 16:
          reinterpret_cast<i16complex&>(in.inputSamples[st][i][pol][0]) = 
            inputSignal<i16complex>(i);
          break;
        default:
          break;
        }
      }

  // Initialize subbands partitioning administration (struct BlockID). We only
  // do the 1st block of whatever.

  // Block number: 0 .. inf
  in.blockID.block = 0;

 // Subband index in the observation: [0, ps.nrSubbands())
  in.blockID.globalSubbandIdx = 0;

  // Subband index for this pipeline/workqueue: [0, subbandIndices.size())
  in.blockID.localSubbandIdx = 0;

  // Subband index for this SubbandProc
  in.blockID.subbandProcSubbandIdx = 0;

  // Initialize delays. We skip delay compensation, but init anyway,
  // so we won't copy uninitialized data to the device.
  for (size_t i = 0; i < in.delaysAtBegin.num_elements(); i++)
    in.delaysAtBegin.get<float>()[i] = 0.0f;
  for (size_t i = 0; i < in.delaysAfterEnd.num_elements(); i++)
    in.delaysAfterEnd.get<float>()[i] = 0.0f;
  for (size_t i = 0; i < in.phaseOffsets.num_elements(); i++)
    in.phaseOffsets.get<float>()[i] = 0.0f;
  for (size_t i = 0; i < in.tabDelays.num_elements(); i++)
    in.tabDelays.get<float>()[i] = 0.0f;

  BeamFormedData out(maxNrTABsPerSAP * nrStokes, nrChannels, nrSamples, ctx);

  for (size_t i = 0; i < out.num_elements(); i++)
    out.get<float>()[i] = 42.0f;

  // Don't bother initializing out.blockID; processSubband() doesn't need it.

  cout << "processSubband()" << endl;
  bwq.processSubband(in, out);
  cout << "processSubband() done" << endl;

  cout << "Output: " << endl;

  // Output verification

  // We can calculate the expected output values, since we're supplying a
  // complex sine/cosine input signal. We only have Stokes-I, so the output
  // should be: (nrStation * amp * scaleFactor * nrChannel1 * nrChannel2)^2
  // - amp is set to the maximum possible value for the bit-mode:
  //   i.e. 127 for 8-bit and 32767 for 16-bit mode
  // - scaleFactor is the scaleFactor applied by the IntToFloat kernel. 
  //   It is 16 for 8-bit mode and 1 for 16-bit mode.
  // Hence, each output sample should be: 
  // - for 16-bit input: (2 * 32767 * 1 * 64 * 64)^2 = 72053196058525696
  // - for 8-bit input: (2 * 127 * 16 * 64 * 64)^2 = 1082398867456

  float outVal;
  switch(nrBitsPerSample) {
  case 8:
    outVal = 
      nrStations * amplitude * scaleFactor * nrChannel1 * nrChannel2 *
      nrStations * amplitude * scaleFactor * nrChannel1 * nrChannel2; 
    break;
  case 16:
    outVal = 
      nrStations * amplitude * scaleFactor * nrChannel1 * nrChannel2 *
      nrStations * amplitude * scaleFactor * nrChannel1 * nrChannel2; 
    break;
  default:
    break;
  }
  cout << "outVal = " << outVal << endl;

  for (size_t s = 0; s < nrStokes; s++)
    for (size_t t = 0; t < nrSamples; t++)
      for (size_t c = 0; c < nrChannels; c++)
        ASSERTSTR(fpEquals(out[s][t][c], outVal), 
                  "out[" << s << "][" << t << "][" << c << "] = " << 
                  out[s][t][c] << "; outVal = " << outVal);
  
  return 0;
}


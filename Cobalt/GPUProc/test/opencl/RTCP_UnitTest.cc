//# RTCP_UnitTest.cc
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
//# $Id: RTCP_UnitTest.cc 25358 2013-06-18 09:05:40Z loose $

#include <lofar_config.h>

#include <iostream>

#include <Common/lofar_complex.h>
#include <Common/LofarLogger.h>
#include <Common/Exception.h>
#include <CoInterface/Parset.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>
#include <GPUProc/opencl/gpu_utils.h>

#include <UnitTest.h>
#include "Kernels/IncoherentStokesTest.h"
#include "Kernels/IntToFloatTest.h"
#include "Kernels/BeamFormerTransposeTest.h"
#include "Kernels/DedispersionChirpTest.h"
#include "Kernels/CoherentStokesTest.h"
//#include "Kernels/UHEP_BeamFormerTest.h"
//#include "Kernels/UHEP_TransposeTest.h"
#include "Kernels/BeamFormerTest.h"
#include "Kernels/CorrelateTriangleTest.h"
//#include "Kernels/UHEP_TriggerTest.h"
#include "Kernels/CorrelateRectangleTest.h"
#include "Kernels/CorrelatorTest.h"
#include "Kernels/FFT_Test.h"
//#include "Kernels/AMD_FFT_Test.h"
#include "Kernels/FIR_FilterTest.h"

//#include  <UnitTest++.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;

// Use our own terminate handler
Exception::TerminateHandler t(Exception::terminate);

int main(int argc, char **argv)
{

  INIT_LOGGER("RTCP");
  std::cout << "running ..." << std::endl;

  if (argc < 2)
  {
    std::cerr << "usage: " << argv[0] << " parset" << std::endl;
    return 1;
  }

  Parset ps(argv[1]);

  std::cout << "Obs ps: nSt=" << ps.nrStations() << " nPol=" << NR_POLARIZATIONS
            << " nSampPerCh=" << ps.nrSamplesPerChannel() << " nChPerSb="
            << ps.nrChannelsPerSubband() << " nTaps=" << ps.nrPPFTaps()
            << " nBitsPerSamp=" << ps.nrBitsPerSample() << std::endl;

  //Correlation unittest
  (FIR_FilterTest)(ps);
  (FFT_Test)(ps);
  //(AMD_FFT_Test)(ps);
  (CorrelatorTest)(ps);
#if defined USE_NEW_CORRELATOR
  (CorrelateRectangleTest)(ps);
  (CorrelateTriangleTest)(ps);
#endif

  // Beamforming unittest
  (IncoherentStokesTest)(ps);
  (IntToFloatTest)(ps);
  (BeamFormerTest)(ps);
  (BeamFormerTransposeTest)(ps);
  (DedispersionChirpTest)(ps);
  (CoherentStokesTest)(ps);

  // UHEP unittest
  //(UHEP_BeamFormerTest)(ps);
  //(UHEP_TransposeTest)(ps);
  //(UHEP_TriggerTest)(ps);

  //return UnitTest::RunAllTests();
  return 0;
}


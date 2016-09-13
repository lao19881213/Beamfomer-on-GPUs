//# UHEP_TransposeKernel.cc
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
//# $Id: UHEP_TransposeKernel.cc 24977 2013-05-21 13:46:16Z amesfoort $

#include <lofar_config.h>

#include "UHEP_TransposeKernel.h"

#include <Common/lofar_complex.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    UHEP_TransposeKernel::UHEP_TransposeKernel(const Parset &ps, cl::Program &program, cl::Buffer &devFFTedData, cl::Buffer &devComplexVoltages, cl::Buffer &devReverseSubbandMapping)
      :
      Kernel(ps, program, "UHEP_Transpose")
    {
      setArg(0, devFFTedData);
      setArg(1, devComplexVoltages);
      setArg(2, devReverseSubbandMapping);

      globalWorkSize = cl::NDRange(256, (ps.nrTABs(0) + 15) / 16, 512 / 16);
      localWorkSize = cl::NDRange(256, 1, 1);

      nrOperations = 0;
      nrBytesRead = (size_t) ps.nrSubbands() * (ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1) * ps.nrTABs(0) * NR_POLARIZATIONS * sizeof(std::complex<float>);
      nrBytesWritten = (size_t) ps.nrTABs(0) * NR_POLARIZATIONS * (ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1) * 512 * sizeof(std::complex<float>);
    }


  }
}


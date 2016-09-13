//# UHEP_InvFFT_Kernel.h
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
//# $Id: UHEP_InvFFT_Kernel.cc 27077 2013-10-24 01:00:58Z amesfoort $

#include <lofar_config.h>

#include "UHEP_InvFFT_Kernel.h"

#include <Common/lofar_complex.h>

#include <GPUProc/global_defines.h>

namespace LOFAR
{
  namespace Cobalt
  {
    UHEP_InvFFT_Kernel::UHEP_InvFFT_Kernel(const Parset &ps, gpu::Module &program, gpu::DeviceMemory &devFFTedData)
      :
      Kernel(ps, program, "inv_fft")
    {
      setArg(0, devFFTedData);
      setArg(1, devFFTedData);

      setEnqueueWorkSizes( gpu::Grid(128, ps.nrTABs(0) * NR_POLARIZATIONS * ps.nrSamplesPerChannel()),
                           gpu::Block(128, 1) );

      size_t nrFFTs = (size_t) ps.nrTABs(0) * NR_POLARIZATIONS * (ps.nrSamplesPerChannel() + NR_STATION_FILTER_TAPS - 1);
      nrOperations = nrFFTs * 5 * 1024 * 10;
      nrBytesRead = nrFFTs * 512 * sizeof(std::complex<float>);
      nrBytesWritten = nrFFTs * 1024 * sizeof(float);
    }

  }
}


//# BeamFormerFactories.h
//#
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
//# $Id: BeamFormerFactories.h 27477 2013-11-21 13:08:20Z loose $

#ifndef LOFAR_GPUPROC_CUDA_BEAM_FORMER_FACTORIES_H
#define LOFAR_GPUPROC_CUDA_BEAM_FORMER_FACTORIES_H

#include <GPUProc/KernelFactory.h>
#include <GPUProc/Kernels/BandPassCorrectionKernel.h>
#include <GPUProc/Kernels/BeamFormerKernel.h>
#include <GPUProc/Kernels/BeamFormerTransposeKernel.h>
#include <GPUProc/Kernels/CoherentStokesKernel.h>
#include <GPUProc/Kernels/DelayAndBandPassKernel.h>
#include <GPUProc/Kernels/FIR_FilterKernel.h>
#include <GPUProc/Kernels/IntToFloatKernel.h>
#include <GPUProc/Kernels/IncoherentStokesKernel.h>
#include <GPUProc/Kernels/IncoherentStokesTransposeKernel.h>

namespace LOFAR
{
  namespace Cobalt
  {
    //# Forward declarations
    class Parset;

    struct BeamFormerFactories
    {
      BeamFormerFactories(const Parset &ps, 
                            size_t nrSubbandsPerSubbandProc = 1);

      KernelFactory<IntToFloatKernel> intToFloat;
      KernelFactory<DelayAndBandPassKernel> delayCompensation;
      KernelFactory<BeamFormerKernel> beamFormer;
      KernelFactory<BeamFormerTransposeKernel> transpose;
      KernelFactory<FIR_FilterKernel> firFilter;
      KernelFactory<CoherentStokesKernel> coherentStokes;
      KernelFactory<IncoherentStokesKernel> incoherentStokes;
      KernelFactory<IncoherentStokesTransposeKernel> incoherentStokesTranspose;
      KernelFactory<FIR_FilterKernel> incoherentFirFilter;
      KernelFactory<BandPassCorrectionKernel> bandPassCorrection;

      BandPassCorrectionKernel::Parameters
      bandPassCorrectionParams(const Parset &ps) const;

      BeamFormerKernel::Parameters
      beamFormerParams(const Parset &ps) const;

      BeamFormerTransposeKernel::Parameters
      transposeParams(const Parset &ps) const;

      CoherentStokesKernel::Parameters
      coherentStokesParams(const Parset &ps) const;

      DelayAndBandPassKernel::Parameters
      delayCompensationParams(const Parset &ps) const;

      FIR_FilterKernel::Parameters
      firFilterParams(const Parset &ps, size_t nrSubbandsPerSubbandProc) const;

      FIR_FilterKernel::Parameters 
      incoherentFirFilterParams(const Parset &ps,
            size_t nrSubbandsPerSubbandProc) const ;

      IncoherentStokesKernel::Parameters 
      incoherentStokesParams(const Parset &ps) const;

      IncoherentStokesTransposeKernel::Parameters 
      incoherentStokesTransposeParams(const Parset &ps) const;


    };

  }
}

#endif

//# BandPassCorrectionKernel.h
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
//# $Id: BandPassCorrectionKernel.h 27477 2013-11-21 13:08:20Z loose $

#ifndef LOFAR_GPUPROC_CUDA_BAND_PASS_CORRECTION_KERNEL_H
#define LOFAR_GPUPROC_CUDA_BAND_PASS_CORRECTION_KERNEL_H

#include <CoInterface/Parset.h>

#include <GPUProc/Kernels/Kernel.h>
#include <GPUProc/KernelFactory.h>
#include <GPUProc/gpu_wrapper.h>

namespace LOFAR
{
  namespace Cobalt
  {

    class BandPassCorrectionKernel : public Kernel
    {
    public:
      static std::string theirSourceFile;
      static std::string theirFunction;

      // Parameters that must be passed to the constructor of the
      // DelayAndBandPassKernel class.
      struct Parameters : Kernel::Parameters
      {
        Parameters(const Parset& ps);
        size_t nrBitsPerSample;
        size_t nrBytesPerComplexSample;
        size_t nrSAPs;
        size_t nrChannels1;
        size_t nrChannels2;
        bool correctBandPass;
      };

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA,
        BAND_PASS_CORRECTION_WEIGHTS
      };

      // Buffers that must be passed to the constructor of the DelayAndBandPassKernel
      // class.
      struct Buffers : Kernel::Buffers
      {
        Buffers(const gpu::DeviceMemory& in, 
                const gpu::DeviceMemory& out,
                const gpu::DeviceMemory& bandPassCorrectionWeights) :
          Kernel::Buffers(in, out), 
          bandPassCorrectionWeights(bandPassCorrectionWeights)
        {}

        gpu::DeviceMemory bandPassCorrectionWeights;
      };

      BandPassCorrectionKernel(const gpu::Stream &stream,
                             const gpu::Module &module,
                             const Buffers &buffers,
                             const Parameters &param);

    };

    //# --------  Template specializations for KernelFactory  -------- #//

    template<> size_t
    KernelFactory<BandPassCorrectionKernel>::bufferSize(BufferType bufferType) const;

    template<> CompileDefinitions
    KernelFactory<BandPassCorrectionKernel>::compileDefinitions() const;
  }
}

#endif


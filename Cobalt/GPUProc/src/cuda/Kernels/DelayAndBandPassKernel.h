//# DelayAndBandPassKernel.h
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
//# $Id: DelayAndBandPassKernel.h 27075 2013-10-23 16:22:29Z amesfoort $

#ifndef LOFAR_GPUPROC_CUDA_DELAY_AND_BAND_PASS_KERNEL_H
#define LOFAR_GPUPROC_CUDA_DELAY_AND_BAND_PASS_KERNEL_H

#include <CoInterface/Parset.h>

#include <GPUProc/Kernels/Kernel.h>
#include <GPUProc/KernelFactory.h>
#include <GPUProc/gpu_wrapper.h>
//#include <GPUProc/PerformanceCounter.h>

namespace LOFAR
{
  namespace Cobalt
  {

    class DelayAndBandPassKernel : public Kernel
    {
    public:
      static std::string theirSourceFile;
      static std::string theirFunction;

      // Parameters that must be passed to the constructor of the
      // DelayAndBandPassKernel class.
      struct Parameters : Kernel::Parameters
      {
        Parameters(const Parset& ps);
        unsigned nrBitsPerSample;
        unsigned nrBytesPerComplexSample;
        unsigned nrSAPs;
        bool delayCompensation;
        bool correctBandPass;
        bool transpose;
        double subbandBandwidth;
      };

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA,
        DELAYS,
        PHASE_OFFSETS,
        BAND_PASS_CORRECTION_WEIGHTS
      };

      // Buffers that must be passed to the constructor of the DelayAndBandPassKernel
      // class.
      struct Buffers : Kernel::Buffers
      {
        Buffers(const gpu::DeviceMemory& in, 
                const gpu::DeviceMemory& out,
                const gpu::DeviceMemory& delaysAtBegin,
                const gpu::DeviceMemory& delaysAfterEnd,
                const gpu::DeviceMemory& phaseOffsets,
                const gpu::DeviceMemory& bandPassCorrectionWeights) :
          Kernel::Buffers(in, out), delaysAtBegin(delaysAtBegin), delaysAfterEnd(delaysAfterEnd), phaseOffsets(phaseOffsets), bandPassCorrectionWeights(bandPassCorrectionWeights)
        {}

        gpu::DeviceMemory delaysAtBegin;
        gpu::DeviceMemory delaysAfterEnd;
        gpu::DeviceMemory phaseOffsets;
        gpu::DeviceMemory bandPassCorrectionWeights;
      };

      DelayAndBandPassKernel(const gpu::Stream &stream,
                             const gpu::Module &module,
                             const Buffers &buffers,
                             const Parameters &param);


      void enqueue(const BlockID &blockId, PerformanceCounter &counter,
                   double subbandFrequency, unsigned SAP);

    };

    //# --------  Template specializations for KernelFactory  -------- #//

    template<> size_t
    KernelFactory<DelayAndBandPassKernel>::bufferSize(BufferType bufferType) const;

    template<> CompileDefinitions
    KernelFactory<DelayAndBandPassKernel>::compileDefinitions() const;
  }
}

#endif


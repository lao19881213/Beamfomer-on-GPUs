//# BeamFormerKernel.h
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
//# $Id: BeamFormerKernel.h 26758 2013-09-30 11:47:21Z loose $

#ifndef LOFAR_GPUPROC_CUDA_BEAM_FORMER_KERNEL_H
#define LOFAR_GPUPROC_CUDA_BEAM_FORMER_KERNEL_H

#include <CoInterface/Parset.h>

#include <GPUProc/Kernels/Kernel.h>
#include <GPUProc/KernelFactory.h>
#include <GPUProc/gpu_wrapper.h>

namespace LOFAR
{
  namespace Cobalt
  {
    class BeamFormerKernel : public Kernel
    {
    public:
      static std::string theirSourceFile;
      static std::string theirFunction;

      // Parameters that must be passed to the constructor of the
      // BeamFormerKernel class.
      struct Parameters : Kernel::Parameters
      {
        Parameters(const Parset& ps);
        unsigned nrSAPs;
        unsigned nrTABs;
        float weightCorrection; // constant weight applied to all weights
        double subbandBandwidth;
      };

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA,
        BEAM_FORMER_DELAYS
      };

      // Buffers that must be passed to the constructor of the BeamFormerKernel
      // class.
      struct Buffers : Kernel::Buffers
      {
        Buffers(const gpu::DeviceMemory& in, 
                const gpu::DeviceMemory& out,
                const gpu::DeviceMemory& beamFormerDelays) :
          Kernel::Buffers(in, out), beamFormerDelays(beamFormerDelays)
        {}

        gpu::DeviceMemory beamFormerDelays;
      };

      BeamFormerKernel(const gpu::Stream &stream,
                             const gpu::Module &module,
                             const Buffers &buffers,
                             const Parameters &param);

      void enqueue(const BlockID &blockId, PerformanceCounter &counter,
                   double subbandFrequency, unsigned SAP);

    };

    //# --------  Template specializations for KernelFactory  -------- #//

    template<> size_t
    KernelFactory<BeamFormerKernel>::bufferSize(BufferType bufferType) const;

    template<> CompileDefinitions
    KernelFactory<BeamFormerKernel>::compileDefinitions() const;
  }
}

#endif


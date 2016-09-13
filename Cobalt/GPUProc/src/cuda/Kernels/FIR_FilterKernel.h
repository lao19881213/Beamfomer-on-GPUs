//# FIR_FilterKernel.h
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
//# $Id: FIR_FilterKernel.h 27075 2013-10-23 16:22:29Z amesfoort $

#ifndef LOFAR_GPUPROC_CUDA_FIR_FILTER_KERNEL_H
#define LOFAR_GPUPROC_CUDA_FIR_FILTER_KERNEL_H

#include <string>
#include <CoInterface/Parset.h>

#include <GPUProc/Kernels/Kernel.h>
#include <GPUProc/KernelFactory.h>
#include <GPUProc/FilterBank.h>
#include <GPUProc/gpu_wrapper.h>

namespace LOFAR
{
  namespace Cobalt
  {
    class FIR_FilterKernel : public Kernel
    {
    public:
      static std::string theirSourceFile;
      static std::string theirFunction;

      // Parameters that must be passed to the constructor of the
      // FIR_FilterKernel class.
      struct Parameters : Kernel::Parameters
      {
        Parameters(const Parset& ps);
        unsigned nrBitsPerSample;
        unsigned nrBytesPerComplexSample;

        // The number of stations or TABs to filter. The FIR filter will
        // deal with either in the same way.
        unsigned nrSTABs;

        // The number of subbands \e this kernel instance will process,
        // typically equal to \c nrSubbandsPerSubbandProc.
        unsigned nrSubbands;

        // The number of PPF filter taps.
        static const unsigned nrTaps = 16;

        // The number of history samples used for each block
        unsigned nrHistorySamples() const;
      };

      enum BufferType
      {
        INPUT_DATA,
        OUTPUT_DATA,
        FILTER_WEIGHTS,
        HISTORY_DATA
      };

      // Buffers that must be passed to the constructor of the FIR_FilterKernel
      // class.
      struct Buffers : Kernel::Buffers
      {
        Buffers(const gpu::DeviceMemory& in, 
                const gpu::DeviceMemory& out,
                const gpu::DeviceMemory& fw,
                const gpu::DeviceMemory& hs) :
          Kernel::Buffers(in, out), filterWeights(fw), historySamples(hs)
        {}
        gpu::DeviceMemory filterWeights;
        gpu::DeviceMemory historySamples;
      };

      FIR_FilterKernel(const gpu::Stream& stream,
                       const gpu::Module& module,
                       const Buffers& buffers,
                       const Parameters& param);

      void enqueue(const BlockID &blockId,
                   PerformanceCounter &counter,
                   unsigned subbandIdx);

      // Put the historyFlags[subbandIdx] in front of the given inputFlags,
      // and update historyFlags[subbandIdx] with the flags of the last samples
      // in inputFlags.
      void prefixHistoryFlags(MultiDimArray<SparseSet<unsigned>, 1> &inputFlags, unsigned subbandIdx);

    private:
      // The Kernel parameters as given to the constructor
      const Parameters params;

      // The flags of the history samples.
      //
      // Dimensions: [nrSubbands][nrStations]
      MultiDimArray<SparseSet<unsigned>, 2> historyFlags;
    };

    //# --------  Template specializations for KernelFactory  -------- #//

    template<> size_t
    KernelFactory<FIR_FilterKernel>::bufferSize(BufferType bufferType) const;

    template<> CompileDefinitions
    KernelFactory<FIR_FilterKernel>::compileDefinitions() const;
  }
}

#endif


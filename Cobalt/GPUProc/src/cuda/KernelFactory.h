//# KernelFactory.h
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
//# $Id: KernelFactory.h 25944 2013-08-06 13:33:07Z klijn $

#ifndef LOFAR_GPUPROC_CUDA_KERNELFACTORY_H
#define LOFAR_GPUPROC_CUDA_KERNELFACTORY_H

#include <string>
#include <CoInterface/Parset.h>
#include <GPUProc/Kernels/Kernel.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>

namespace LOFAR
{
  namespace Cobalt
  {
    // Abstract base class of the templated KernelFactory class.
    class KernelFactoryBase
    {
    public:
      // Pure virtual destructor, because this is an abstract base class.
      virtual ~KernelFactoryBase() = 0;

    protected:
      // Return compile definitions to use when creating PTX code for any
      // Kernel.
      CompileDefinitions
      compileDefinitions(const Kernel::Parameters& param) const;

      // Return compile flags to use when creating PTX code for any Kernel.
      CompileFlags
      compileFlags(const Kernel::Parameters& param) const;
    };

    // Declaration of a generic factory class. For each concrete Kernel class
    // (e.g. FIR_FilterKernel), a specialization must exist of the constructor
    // and of the bufferSize() method.
    template<typename T> class KernelFactory : public KernelFactoryBase
    {
    public:
      // typedef typename T::Parameters Parameters;
      typedef typename T::BufferType BufferType;
      typedef typename T::Buffers Buffers;

      // Construct a factory for creating Kernel objects of type \c T, using the
      // settings in the Parset \a ps.
      KernelFactory(const Parset& ps) :
        itsParameters(ps),
        itsPTX(createPTX(T::theirSourceFile,
                         compileDefinitions(),
                         compileFlags()))
      {
      }

      // Construct a factory for creating Kernel objects of type \c T, using the
      // settings provided by \a params.
      KernelFactory(const typename T::Parameters &params) :
        itsParameters(params),
        itsPTX(_createPTX())
      {
      }

      // Create a new Kernel object of type \c T.
      T* create(const gpu::Stream& stream,
                const typename T::Buffers& buffers) const
      {
        return new T(
          stream, createModule(stream.getContext(), 
                               T::theirSourceFile,
                               itsPTX), 
          buffers, itsParameters);
      }

      // // Create a new Kernel object of type \c T, using kernel-specific
      // // parameters to instantiate this new object.
      // T* create(const Parameters& params)
      //   { return new T(params); }

      // Return required buffer size for \a bufferType
      size_t bufferSize(BufferType bufferType) const;

    private:
      // Used by the constructors to construct the PTX from the other
      // members.
      std::string _createPTX() const {
        return createPTX(T::theirSourceFile,
                           compileDefinitions(),
                           compileFlags());
      }

      // Return compile definitions to use when creating PTX code for kernels of
      // type \c T, using the parameters stored in \c itsParameters.
      CompileDefinitions compileDefinitions() const {
        return KernelFactoryBase::compileDefinitions(itsParameters);
      }

      // Return compile flags to use when creating PTX code for kernels of type
      // \c T.
      CompileFlags compileFlags() const {
        return KernelFactoryBase::compileFlags(itsParameters);
      }

      // Additional parameters needed to create a Kernel object of type \c T.
      typename T::Parameters itsParameters;

      // PTX code, generated for kernels of type \c T, using information in the
      // Parset that was passed to the constructor.
      std::string itsPTX;
    };

  } // namespace Cobalt

} // namespace LOFAR

#endif

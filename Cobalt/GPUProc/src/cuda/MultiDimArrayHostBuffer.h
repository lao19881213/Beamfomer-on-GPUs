//# MultiDimArrayHostBuffer.h
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
//# $Id: MultiDimArrayHostBuffer.h 26249 2013-08-27 22:08:35Z amesfoort $

#ifndef LOFAR_GPUPROC_CUDA_MULTI_DIM_ARRAY_HOST_BUFFER_H
#define LOFAR_GPUPROC_CUDA_MULTI_DIM_ARRAY_HOST_BUFFER_H

#include <CoInterface/MultiDimArray.h>

#include "gpu_wrapper.h"

namespace LOFAR
{
  namespace Cobalt
  {

    // A MultiDimArray allocated as a HostBuffer
    // Note: Elements are not constructed/destructed.
    template <typename T, unsigned DIM>
    class MultiDimArrayHostBuffer : public gpu::HostMemory,
                                    public MultiDimArray<T, DIM>
    {
    public:
      template <typename ExtentList>
      MultiDimArrayHostBuffer(const ExtentList &extents, const gpu::Context &context,
                              unsigned int flags = 0)
      :
        HostMemory(context, MultiDimArray<T, DIM>::nrElements(extents) * sizeof(T), flags),
        MultiDimArray<T, DIM>(extents, gpu::HostMemory::get<T>(), false)
      {
      }

      using HostMemory::size;

    private:
      MultiDimArrayHostBuffer(); // don't use
      MultiDimArrayHostBuffer(const MultiDimArrayHostBuffer<T, DIM> &rhs); // don't use
      MultiDimArrayHostBuffer<T, DIM> &operator=(const MultiDimArrayHostBuffer<T, DIM> &rhs); // don't use
      using MultiDimArray<T, DIM>::resize; // don't use
    };

  } // namespace Cobalt
} // namespace LOFAR

#endif


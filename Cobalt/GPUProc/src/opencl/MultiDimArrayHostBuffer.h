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
//# $Id: MultiDimArrayHostBuffer.h 25199 2013-06-05 23:46:56Z amesfoort $

#ifndef LOFAR_GPUPROC_OPENCL_MULTI_DIM_ARRAY_HOST_BUFFER_H
#define LOFAR_GPUPROC_OPENCL_MULTI_DIM_ARRAY_HOST_BUFFER_H

#include <CoInterface/Allocator.h>
#include <CoInterface/MultiDimArray.h>

#include "gpu_incl.h"

namespace LOFAR
{
  namespace Cobalt
  {

    // A buffer on the GPU (device), to which CPU (host) buffers can be attached.
    class DeviceBuffer
    {
    public:
      DeviceBuffer( cl::CommandQueue &queue, cl_mem_flags flags, size_t size )
      :
        size(size),
        queue(queue),
        flags(flags),
        buffer(queue.getInfo<CL_QUEUE_CONTEXT>(), flags | CL_MEM_ALLOC_HOST_PTR, size)
      {
      }

      const size_t size;
      cl::CommandQueue &queue;
      const cl_mem_flags flags;
      cl::Buffer buffer;

      // Stores transfer information
      cl::Event event;

      operator cl::Buffer & ()
      {
        return buffer;
      }

      // Copies data to the GPU
      void hostToDevice(void *ptr, size_t size, bool synchronous = false)
      {
        ASSERT(size <= this->size);

        queue.enqueueWriteBuffer(buffer, synchronous ? CL_TRUE : CL_FALSE, 0, size, ptr, 0, &event);
      }

      // Copies data from the GPU
      void deviceToHost(void *ptr, size_t size, bool synchronous = false)
      {
        ASSERT(size <= this->size);

        queue.enqueueReadBuffer(buffer, synchronous ? CL_TRUE : CL_FALSE, 0, size, ptr, 0, &event);
      }

      // Allocates a buffer for transfer with the GPU
      void *allocateHostBuffer( size_t size, cl_mem_flags hostBufferFlags = CL_MEM_READ_WRITE )
      {
        ASSERT(size <= this->size);

        return queue.enqueueMapBuffer(buffer, CL_TRUE, map_flags(hostBufferFlags), 0, size);
      }

      // Deallocates a buffer for transfer with the GPU
      void deallocateHostBuffer( void *ptr )
      {
        queue.enqueueUnmapMemObject(buffer, ptr);
      }

    private:
      // Can't copy cl::Buffer
      DeviceBuffer(const DeviceBuffer &other);

      // Convert the cl_mem_flags to cl_map_flags
      static cl_map_flags map_flags(cl_mem_flags flags) {
        return flags & CL_MEM_READ_WRITE ? CL_MAP_READ | CL_MAP_WRITE
             : flags & CL_MEM_READ_ONLY  ? CL_MAP_READ
             : flags & CL_MEM_WRITE_ONLY ? CL_MAP_WRITE
             : 0;
      }
    };

    // A buffer on the CPU (host), attached to a buffer on the GPU (device)
    class HostBuffer
    {
    public:
      HostBuffer( DeviceBuffer &deviceBuffer, size_t size, cl_mem_flags hostBufferFlags = CL_MEM_READ_WRITE )
      :
        deviceBuffer(deviceBuffer),
        ptr(deviceBuffer.allocateHostBuffer(size, hostBufferFlags)),
        size(size)
      {
      }

      ~HostBuffer()
      {
        deviceBuffer.deallocateHostBuffer(ptr);
      }

      void hostToDevice(bool synchronous = false)
      {
        deviceBuffer.hostToDevice(ptr, size, synchronous);
      }

      void deviceToHost(bool synchronous = false)
      {
        deviceBuffer.deviceToHost(ptr, size, synchronous);
      }

      DeviceBuffer &deviceBuffer;

      operator cl::Buffer& () {
        return deviceBuffer;
      }

    protected:
      void * const ptr;
      const size_t size;

    private:
      // Copying is expensive (requires allocation),
      // so forbid it to prevent accidental copying.
      HostBuffer(const HostBuffer &other);
    };

    // A MultiDimArray allocated as a HostBuffer
    template <typename T, size_t DIM>
    class MultiArrayHostBuffer : public HostBuffer, public MultiDimArray<T, DIM>
    {
    public:
      template <typename ExtentList>
      MultiArrayHostBuffer(const ExtentList &extents, cl_mem_flags hostBufferFlags, DeviceBuffer &deviceBuffer)
      :
        HostBuffer(deviceBuffer, this->nrElements(extents) * sizeof(T), hostBufferFlags),
        MultiDimArray<T,DIM>(extents, static_cast<T*>(ptr), true)
      {
      }

      size_t bytesize() const
      {
        return this->num_elements() * sizeof(T);
      }
    };

    // A 1:1 buffer on CPU and GPU
    template <typename T, size_t DIM>
    class MultiArraySharedBuffer : public DeviceBuffer, public MultiArrayHostBuffer<T, DIM>
    {
    public:
      template <typename ExtentList>
      MultiArraySharedBuffer(const ExtentList &extents, cl::CommandQueue &queue, cl_mem_flags hostBufferFlags = CL_MEM_READ_WRITE, cl_mem_flags deviceBufferFlags = CL_MEM_READ_WRITE)
        :
        DeviceBuffer(queue, deviceBufferFlags, this->nrElements(extents) * sizeof(T)),
        MultiArrayHostBuffer<T, DIM>(extents, hostBufferFlags, *this)
      {
      }

      // Select the desired interface
      using HostBuffer::hostToDevice;
      using HostBuffer::deviceToHost;
      using DeviceBuffer::operator cl::Buffer&;
    };

  } // namespace Cobalt
} // namespace LOFAR

#endif


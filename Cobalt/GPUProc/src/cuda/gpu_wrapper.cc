//# gpu_wrapper.cc: CUDA-specific wrapper classes for GPU types.
//#
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: gpu_wrapper.cc 27248 2013-11-05 17:37:41Z amesfoort $

#include <lofar_config.h>

#include "gpu_wrapper.h"

#include <string>
#include <iostream>
#include <algorithm>  // for std::min and std::max

#include <boost/noncopyable.hpp>
#include <boost/format.hpp>

#include <Common/Exception.h>
#include <Common/LofarLogger.h>

#include <GPUProc/global_defines.h>
#include <GPUProc/PerformanceCounter.h>

// Convenience macro to call a CUDA Device API function and throw a
// CUDAException if an error occurred.
#define checkCuCall(func)                                               \
  do {                                                                  \
    CUresult result = func;                                             \
    if (result != CUDA_SUCCESS) {                                       \
      LOG_ERROR_STR(                                                    \
             # func << ": " << LOFAR::Cobalt::gpu::errorMessage(result)); \
      THROW (LOFAR::Cobalt::gpu::CUDAException,                         \
             # func << ": " << LOFAR::Cobalt::gpu::errorMessage(result)); \
    }                                                                   \
  } while(0)

LOFAR::Exception::TerminateHandler th(LOFAR::Exception::terminate);

using boost::format;

namespace LOFAR
{
  namespace Cobalt
  {
    namespace gpu
    {

      std::string cufftErrorMessage(cufftResult errcode)
      {
        switch (errcode) {
        case CUFFT_SUCCESS:
          return "Success";
        case CUFFT_INVALID_PLAN:
          return "Invalid plan";
        case CUFFT_ALLOC_FAILED:
          return "Failed to allocate CPU or GPU memory";
        case CUFFT_INVALID_TYPE: // no longer used
          return "Invalid type";
        case CUFFT_INVALID_VALUE:
          return "Invalid pointer or parameter";
        case CUFFT_INTERNAL_ERROR:
          return "Driver or internal cuFFT error";
        case CUFFT_EXEC_FAILED:
          return "Failed to execute FFT kernel";
        case CUFFT_SETUP_FAILED:
          return "Failed to initialise cuFFT library";
        case CUFFT_INVALID_SIZE:
          return "Invalid transform size";
        case CUFFT_UNALIGNED_DATA: // no longer used
          return "Data not properly aligned";
        default:
          std::stringstream str;
          str << "Unknown error (" << errcode << ")";
          return str.str();
        }
      }

      std::string errorMessage(CUresult errcode)
      {
        switch (errcode) {
        case CUDA_SUCCESS:
          return "Success";
        case CUDA_ERROR_INVALID_VALUE:
          return "Invalid value";
        case CUDA_ERROR_OUT_OF_MEMORY:
          return "Out of memory";
        case CUDA_ERROR_NOT_INITIALIZED:
          return "Not initialized";
        case CUDA_ERROR_DEINITIALIZED:
          return "Deinitialized";
        case CUDA_ERROR_PROFILER_DISABLED:
          return "Profiler disabled";
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
          return "Profiler not initialized";
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
          return "Profiler already started";
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
          return "Profiler already stopped";
        case CUDA_ERROR_NO_DEVICE:
          return "No device";
        case CUDA_ERROR_INVALID_DEVICE:
          return "Invalid device";
        case CUDA_ERROR_INVALID_IMAGE:
          return "Invalid image";
        case CUDA_ERROR_INVALID_CONTEXT:
          return "Invalid context";
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
          return "Context already current";
        case CUDA_ERROR_MAP_FAILED:
          return "Map failed";
        case CUDA_ERROR_UNMAP_FAILED:
          return "Unmap failed";
        case CUDA_ERROR_ARRAY_IS_MAPPED:
          return "Array is mapped";
        case CUDA_ERROR_ALREADY_MAPPED:
          return "Already mapped";
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
          return "No binary for GPU";
        case CUDA_ERROR_ALREADY_ACQUIRED:
          return "Already acquired";
        case CUDA_ERROR_NOT_MAPPED:
          return "Not mapped";
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
          return "Not mapped as array";
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
          return "Not mapped as pointer";
        case CUDA_ERROR_ECC_UNCORRECTABLE:
          return "ECC uncorrectable";
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
          return "Unsupported limit";
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
          return "Context already in use";
        case CUDA_ERROR_INVALID_SOURCE:
          return "Invalid source";
        case CUDA_ERROR_FILE_NOT_FOUND:
          return "File not found";
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
          return "Shared object symbol not found";
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
          return "Shared object init failed";
        case CUDA_ERROR_OPERATING_SYSTEM:
          return "Operating system";
        case CUDA_ERROR_INVALID_HANDLE:
          return "Invalid handle";
        case CUDA_ERROR_NOT_FOUND:
          return "Not found";
        case CUDA_ERROR_NOT_READY:
          return "Not ready";
        case CUDA_ERROR_LAUNCH_FAILED:
          return "Launch failed";
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
          return "Launch out of resources";
        case CUDA_ERROR_LAUNCH_TIMEOUT:
          return "Launch timeout";
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
          return "Launch incompatible texturing";
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
          return "Peer access already enabled";
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
          return "Peer access not enabled";
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
          return "Primary context active";
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
          return "Context is destroyed";
        case CUDA_ERROR_UNKNOWN:
          return "Unknown";
#if CUDA_VERSION >= 4010
        case CUDA_ERROR_ASSERT:
          return "Assert";
        case CUDA_ERROR_TOO_MANY_PEERS:
          return "Too many peers";
        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
          return "Host memory already registered";
        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
          return "Host memory not registered";
#endif
#if CUDA_VERSION >= 5000
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
          return "Peer access unsupported";
        case CUDA_ERROR_NOT_PERMITTED:
          return "Not permitted";
        case CUDA_ERROR_NOT_SUPPORTED:
          return "Not supported";
#endif
        default:
          std::stringstream str;
          str << "Unknown error (" << errcode << ")";
          return str.str();
        }
      }


      Block::Block(unsigned int x_, unsigned int y_, unsigned int z_) :
        x(x_), y(y_), z(z_)
      {
      }

      std::ostream& operator<<(std::ostream& os, const Block& block)
      {
        os << "[" << block.x << ", " << block.y << ", " << block.z << "]";
        return os;
      }

      Grid::Grid(unsigned int x_, unsigned int y_, unsigned int z_) :
        x(x_), y(y_), z(z_)
      {
      }

      std::ostream& operator<<(std::ostream& os, const Grid& grid)
      {
        os << "[" << grid.x << ", " << grid.y << ", " << grid.z << "]";
        return os;
      }

      Platform::Platform(unsigned int flags)
      {
        // cuInit() is thread-safe, so we don't have to mutex it.
        // In fact, if you start with multiple threads, all threads that
        // do CUDA calls must first call cuInit().
        checkCuCall(cuInit(flags));
      }

      int Platform::version() const
      {
        int version;
        checkCuCall(cuDriverGetVersion(&version));
        return version;
      }

      size_t Platform::size() const
      {
        int nrDevices;
        checkCuCall(cuDeviceGetCount(&nrDevices));
        return (size_t)nrDevices;
      }

      std::vector<Device> Platform::devices() const
      {
        std::vector<Device> devices;

        size_t nrDevices = size();

        for (size_t i = 0; i < nrDevices; ++i) {
          devices.push_back(Device(i));
        }

        // sort to get a predictable order,
        // because CUDA derives its own sorting
        // based on expected performance, which
        // might differ per NUMA binding.
        sort(devices.begin(), devices.end());

        return devices;
      }

      std::string Platform::getName() const
      {
        return "NVIDIA CUDA";
      }

      unsigned Platform::getMaxThreadsPerBlock() const
      {
        const std::vector<Device> _devices = devices();

        unsigned lowest = 0;

        for (std::vector<Device>::const_iterator i = _devices.begin(); i != _devices.end(); ++i) {
          const unsigned maxThreadsPerBlock = i->getMaxThreadsPerBlock();

          if (i == _devices.begin() || maxThreadsPerBlock < lowest)
            lowest = maxThreadsPerBlock;
        }

        return lowest;
      }


      Device::Device(int ordinal)
      {
        checkCuCall(cuDeviceGet(&_device, ordinal));
      }

      bool Device::operator<(const Device &other) const
      {
        return pciId() < other.pciId();
      }

      std::string Device::getName() const
      {
        char name[1024];

        // NV ref is not crystal clear on returned str len. Better be safe
        // and reserve an extra byte for the \0 terminator.
        checkCuCall(cuDeviceGetName(name, sizeof name - 1, _device));
        return std::string(name);
      }

      unsigned Device::getComputeCapabilityMajor() const
      {
#if CUDA_VERSION >= 5000
        return (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
#else
        int major;
        int minor;

        checkCuCall(cuDeviceComputeCapability(&major, &minor, _device));

        return major;
#endif
      }

      unsigned Device::getComputeCapabilityMinor() const
      {
#if CUDA_VERSION >= 5000
        return (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
#else
        int major;
        int minor;

        checkCuCall(cuDeviceComputeCapability(&major, &minor, _device));

        return minor;
#endif
      }

      size_t Device::getTotalGlobalMem() const
      {
        size_t value;

        checkCuCall(cuDeviceTotalMem(&value, _device));
        return value;
      }

      size_t Device::getBlockSharedMem() const
      {
        return (size_t)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
      }

      size_t Device::getTotalConstMem() const
      {
        return (size_t)getAttribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
      }

      std::string Device::pciId() const
      {
        int bus    = getAttribute(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
        int device = getAttribute(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);

        return str(format("%04x:%04x") % bus % device);
      }

      unsigned Device::getMaxThreadsPerBlock() const
      {
        return (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
      }

      struct Block Device::getMaxBlockDims() const
      {
        Block block;
        block.x = (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        block.y = (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        block.z = (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        return block;
      }

      struct Grid Device::getMaxGridDims() const
      {
        Grid grid;
        grid.x = (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        grid.y = (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        grid.z = (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
        return grid;
      }

      unsigned Device::getMultiProcessorCount() const
      {
        return (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
      }

      unsigned Device::getMaxThreadsPerMultiProcessor() const
      {
        return (unsigned)getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
      }

      int Device::getAttribute(CUdevice_attribute attribute) const
      {
        int value;

        checkCuCall(cuDeviceGetAttribute(&value, attribute, _device));
        return value;
      }

      class Context::Impl : boost::noncopyable
      {
      public:
        Impl(CUdevice device, unsigned int flags)
        {
          checkCuCall(cuCtxCreate(&_context, flags, device));

          // Make the context floating, so we can tie it to any thread
          freeCurrent();
        }

        ~Impl()
        {
          checkCuCall(cuCtxDestroy(_context));
        }

        CUdevice getDevice() const
        {
          CUdevice dev;
          checkCuCall(cuCtxGetDevice(&dev));
          return dev;
        }

        void setCacheConfig(CUfunc_cache config) const
        {
          checkCuCall(cuCtxSetCacheConfig(config));
        }

        void setSharedMemConfig(CUsharedconfig config) const
        {
#if CUDA_VERSION >= 4020
          checkCuCall(cuCtxSetSharedMemConfig(config));
#else
          (void)config;
#endif
        }

        void setCurrent() const
        {
          checkCuCall(cuCtxPushCurrent(_context));
        }

        void freeCurrent() const
        {
          checkCuCall(cuCtxPopCurrent(NULL));
        }

      private:
        CUcontext _context;
      };

      Context::Context(const Device &device, unsigned int flags) :
        _impl(new Impl(device._device, flags))
      {
      }

      Device Context::getDevice() const
      {
        ScopedCurrentContext scc(*this);

        return Device(_impl->getDevice());
      }

      void Context::setCacheConfig(CUfunc_cache config) const
      {
        ScopedCurrentContext scc(*this);

        _impl->setCacheConfig(config);
      }

      void Context::setSharedMemConfig(CUsharedconfig config) const
      {
        ScopedCurrentContext scc(*this);

        _impl->setSharedMemConfig(config);
      }


      ScopedCurrentContext::ScopedCurrentContext(const Context &context):
        _context(context)
      {
        _context._impl->setCurrent();
      }

      ScopedCurrentContext::~ScopedCurrentContext()
      {
        _context._impl->freeCurrent();
      }


      class HostMemory::Impl : boost::noncopyable
      {
      public:
        Impl(const Context &context, size_t size, unsigned int flags):
          _context(context),
          _size(size)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemHostAlloc(&_ptr, size, flags));
        }

        ~Impl()
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemFreeHost(_ptr));
        }

        void *get() const
        {
          return _ptr;
        }

        size_t size() const
        {
          return _size;
        }

      private:
        const Context _context;

        void *_ptr;
        size_t _size;
      };

      HostMemory::HostMemory(const Context &context, size_t size, unsigned int flags) :
        _impl(new Impl(context, size, flags))
      {
      }

      size_t HostMemory::size() const
      {
        return _impl->size();
      }

      void* HostMemory::getPtr() const
      {
        return _impl->get();
      }


      class DeviceMemory::Impl : boost::noncopyable
      {
      public:
        Impl(const Context &context, size_t size):
          _context(context),
          _size(size)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemAlloc(&_ptr, std::max(1UL, size)));
        }

        ~Impl()
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemFree(_ptr));
        }

        CUdeviceptr get() const
        {
          return _ptr;
        }

        void set(unsigned char uc, size_t n)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemsetD8(_ptr, uc, n));
        }

        size_t size() const
        {
          return _size;
        }

        Context getContext() const
        {
          return _context;
        }

      //private: // Functions needs its address to set kernel args
        CUdeviceptr _ptr;
      private:
        const Context _context;
        size_t _size;
      };

      DeviceMemory::DeviceMemory(const Context &context, size_t size) :
        _impl(new Impl(context, size))
      {
      }

      void *DeviceMemory::get() const
      {
        return (void *)_impl->get();
      }

      void DeviceMemory::set(unsigned char uc, size_t n)
      {
        _impl->set(uc, std::min(n, size()));
      }

      size_t DeviceMemory::size() const
      {
        return _impl->size();
      }

      HostMemory DeviceMemory::fetch() const
      {
        // Create a host buffer of the right size
        // in the right context.
        HostMemory host(_impl->getContext(), size());

        // Read the contents of our buffer synchronously,
        // using a dedicated stream.
        Stream s(_impl->getContext());
        s.readBuffer(host, *this, true);

        return host;
      }


      class Module::Impl : boost::noncopyable
      {
      public:
        Impl(): _context(0), _module(0)
        {
        }

        Impl(const Context &context, const std::string &fname):
          _context(context)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuModuleLoad(&_module, fname.c_str()));
        }

        Impl(const Context &context, const void *image):
          _context(context)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuModuleLoadData(&_module, image));
        }

        Impl(const Context &context, const void *image, Module::optionmap_t &options):
          _context(context)
        {
          // Convert our option map to two arrays for CUDA
          std::vector<CUjit_option> keys;
          std::vector<void*> values;

          for (optionmap_t::const_iterator i = options.begin(); i != options.end(); ++i) {
            keys.push_back(i->first);
            values.push_back(i->second);
          }

          ScopedCurrentContext scc(_context);

          checkCuCall(cuModuleLoadDataEx(&_module, image, options.size(),
                                         &keys[0], &values[0]));

          for (size_t i = 0; i < keys.size(); ++i) {
            options[keys[i]] = values[i];
          }

        }

        ~Impl()
        {
          if (_module) {
            ScopedCurrentContext scc(_context);

            checkCuCall(cuModuleUnload(_module));
          }
        }

        Context getContext() const
        {
          return _context;
        }

      private:
        const Context _context;

        CUmodule _module; // NOTE: can be 0

        friend class Function;
      };

      Module::Module() :
        _impl(new Impl())
      {
      }

      Module::Module(const Context &context, const std::string &fname) :
        _impl(new Impl(context, fname))
      {
      }

      Module::Module(const Context &context, const void *image) :
        _impl(new Impl(context, image))
      {
      }

      Module::Module(const Context &context, const void *image, optionmap_t &options):
        _impl(new Impl(context, image, options))
      {
      }

      Context Module::getContext() const
      {
        return _impl->getContext();
      }


      Function::Function(const Module &module, const std::string &name):
        _context(module.getContext()),
        _module(module),
        _name(name)
      {
        ScopedCurrentContext scc(_context);

        checkCuCall(cuModuleGetFunction(&_function, module._impl->_module,
                                        name.c_str()));
      }

      std::string Function::name() const
      {
        return _name;
      }

      void Function::setArg(size_t index, const DeviceMemory &mem)
      {
        doSetArg(index, &mem._impl->_ptr);
      }

      void Function::setArg(size_t index, const void **val)
      {
        doSetArg(index, (const void *)val);
      }

      void Function::doSetArg(size_t index, const void *argp)
      {
        if (index >= _kernelArgs.size()) {
          _kernelArgs.resize(index + 1);
        }
        _kernelArgs[index] = argp;
      }

      int Function::getAttribute(CUfunction_attribute attribute) const
      {
        ScopedCurrentContext scc(_context);

        int value;
        checkCuCall(cuFuncGetAttribute(&value, attribute, _function));
        return value;
      }

      void Function::setSharedMemConfig(CUsharedconfig config) const
      {
#if CUDA_VERSION >= 4020
        ScopedCurrentContext scc(_context);

        checkCuCall(cuFuncSetSharedMemConfig(_function, config));
#else
        (void)config;
#endif
      }


      class Event::Impl : boost::noncopyable
      {
      public:
        Impl(const Context &context, unsigned int flags):
          _context(context)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuEventCreate(&_event, flags));
        }

        ~Impl()
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuEventDestroy(_event));
        }

        float elapsedTime(CUevent other) const
        {
          ScopedCurrentContext scc(_context);

          float ms;
          checkCuCall(cuEventElapsedTime(&ms, other, _event));
          return ms;
        }

        void wait()
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuEventSynchronize(_event));
        }

      //private: // Stream needs it to wait for and record events
        CUevent _event;
      private:
        const Context _context;
      };

      Event::Event(const Context &context, unsigned int flags):
        _impl(new Impl(context, flags))
      {
      }

      float Event::elapsedTime(Event &second) const
      {
        return _impl->elapsedTime(second._impl->_event);
      }

      void Event::wait()
      {
        _impl->wait();
      }


      class Stream::Impl : boost::noncopyable
      {
      public:
        Impl(const Context &context, unsigned int flags):
          _context(context)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuStreamCreate(&_stream, flags));
        }

        ~Impl()
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuStreamDestroy(_stream));
        }

        void memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr, 
                             size_t size)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemcpyHtoDAsync(devPtr, hostPtr, size, _stream));
        }

        void memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemcpyDtoHAsync(hostPtr, devPtr, size, _stream));
        }

        void memcpyDtoDAsync(CUdeviceptr targetPtr, CUdeviceptr sourcePtr, 
                             size_t size)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuMemcpyDtoDAsync(targetPtr, sourcePtr, size, _stream));
        }

        void launchKernel(CUfunction function, unsigned gridX, unsigned gridY,
                          unsigned gridZ, unsigned blockX, unsigned blockY,
                          unsigned blockZ, unsigned sharedMemBytes,
                          void **parameters)
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuLaunchKernel(function, gridX, gridY, gridZ, blockX,
                                     blockY, blockZ, sharedMemBytes, _stream,
                                     parameters, NULL));
        }

        bool query() const
        {
          ScopedCurrentContext scc(_context);

          CUresult rv = cuStreamQuery(_stream);

          switch (rv) {
            case CUDA_ERROR_NOT_READY:
              return false;

            case CUDA_SUCCESS:
              return true;

            default:
              checkCuCall(rv); // throws

              ASSERT(false); // not reached; silence compilation warning
          }
        }

        void synchronize() const
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuStreamSynchronize(_stream));
        }

        void waitEvent(CUevent event) const
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuStreamWaitEvent(_stream, event, 0));
        }

        void recordEvent(CUevent event) const
        {
          ScopedCurrentContext scc(_context);

          checkCuCall(cuEventRecord(event, _stream));
        }

        CUstream get() const
        {
          return _stream;
        }

        Context getContext() const
        {
          return _context;
        }

      private:
        const Context _context;
        CUstream _stream;
      };

      Stream::Stream(const Context &context, unsigned int flags):
        _impl(new Impl(context, flags)),
        force_synchronous(profiling) // TODO: properly set this based on something else
      {
      }

      void Stream::writeBuffer(const DeviceMemory &devMem, 
                               const HostMemory &hostMem,
                               bool synchronous) const
      {
        // tmp check: avoid async writeBuffer request that will fail later.
        // TODO: This interface may still change at which point a cleaner solution can be used.
        if (hostMem.size() > devMem.size())
        {
          THROW(CUDAException, "writeBuffer(): host buffer too large for device buffer: host buffer is " << hostMem.size() << " bytes, device buffer is " << devMem.size() << " bytes");
        }

        _impl->memcpyHtoDAsync((CUdeviceptr)devMem.get(), 
                               hostMem.get<void>(),
                               hostMem.size());
        if (synchronous || force_synchronous) 
        {
          synchronize();
        }
      }

      void Stream::writeBuffer(const DeviceMemory &devMem, const HostMemory &hostMem,
                         const PerformanceCounter &counter, bool synchronous) const
      {
        if (gpuProfiling)
        {
          recordEvent(counter.start);
          writeBuffer(devMem, hostMem, synchronous); 
          recordEvent(counter.stop);
        }
        else
        {
          writeBuffer(devMem, hostMem, synchronous);
        }
      }

      void Stream::copyBuffer(const DeviceMemory &devTarget, 
                              const DeviceMemory &devSource,
                              bool synchronous) const
      {
        // tmp check: avoid async writeBuffer request that will fail later.
        // TODO: This interface may still change at which point a cleaner solution can be used.
        if (devSource.size() > devTarget.size())
        {
          THROW(CUDAException, "copyBuffer(): device source buffer too large for device target buffer: " <<
                "source buffer is " << devSource.size() << " bytes, " << 
                "device buffer is " << devTarget.size() << " bytes");
        }

        _impl->memcpyDtoDAsync((CUdeviceptr)devTarget.get(), 
                               (CUdeviceptr)devSource.get(),
                               devSource.size());
        if (synchronous || force_synchronous) 
        {
          synchronize();
        }
      }

      void Stream::copyBuffer(const DeviceMemory &devTarget, 
                              const DeviceMemory &devSource,
                              const PerformanceCounter &counter,
                              bool synchronous) const
      {
        if (gpuProfiling)
        {
          recordEvent(counter.start);
          copyBuffer(devTarget, devSource, synchronous); 
          recordEvent(counter.stop);
        }
        else
        {
          copyBuffer(devTarget, devSource, synchronous);
        }
      }

      void Stream::readBuffer(const HostMemory &hostMem, 
                              const DeviceMemory &devMem,
                              bool synchronous) const
      {
        // Host buffer can be smaller, because the device
        // buffers can be used for multiple purposes in
        // the CUDA code, and thus can be larger than
        // needed here.
        size_t size = std::min(devMem.size(), hostMem.size());

        _impl->memcpyDtoHAsync(hostMem.get<void>(),
                               (CUdeviceptr)devMem.get(),
                               size);
        if (synchronous || force_synchronous) 
        {
          synchronize();
        }
      }

      void Stream::readBuffer(const HostMemory &hostMem, const DeviceMemory &devMem,
                        const PerformanceCounter &counter, bool synchronous) const
      {
        if (gpuProfiling)
        {
          recordEvent(counter.start);
          readBuffer(hostMem, devMem, synchronous);  
          recordEvent(counter.stop);
        }
        else
        {
          writeBuffer(devMem, hostMem, synchronous);
        }
      }


      void Stream::launchKernel(const Function &function,
                                const Grid &grid, const Block &block) const
      {
        LOG_DEBUG_STR("Launching " << function._name);

        const unsigned dynSharedMemBytes = 0; // we don't need this for LOFAR
        _impl->launchKernel(function._function, grid.x, grid.y, grid.z,
                            block.x, block.y, block.z, dynSharedMemBytes,
                            const_cast<void **>(&function._kernelArgs[0]));

        if (force_synchronous) {
          synchronize();
        }
      }

      bool Stream::query() const
      {
        return _impl->query();
      }

      void Stream::synchronize() const
      {
        _impl->synchronize();
      }

      void Stream::waitEvent(const Event &event) const
      {
        _impl->waitEvent(event._impl->_event);
      }

      void Stream::recordEvent(const Event &event) const
      {
        _impl->recordEvent(event._impl->_event);
      }

      CUstream Stream::get() const
      {
        return _impl->get();
      }

      Context Stream::getContext() const
      {
        return _impl->getContext();
      }

      bool Stream::isSynchronous() const
      {
        return force_synchronous;
      }


    } // namespace gpu
  } // namespace Cobalt
} // namespace LOFAR


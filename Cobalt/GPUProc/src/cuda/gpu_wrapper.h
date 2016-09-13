//# gpu_wrapper.h: CUDA-specific wrapper classes for GPU types.
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
//# $Id: gpu_wrapper.h 27077 2013-10-24 01:00:58Z amesfoort $

#ifndef LOFAR_GPUPROC_CUDA_GPU_WRAPPER_H
#define LOFAR_GPUPROC_CUDA_GPU_WRAPPER_H

// \file cuda/gpu_wrapper.h
// C++ wrappers for CUDA akin the OpenCL C++ wrappers.
// Uses the "Pimpl" idiom for resource managing classes (i.e. that need to
// control copying having a non-trivial destructor. For more info on Pimpl, see
// http://www.boost.org/doc/libs/release/libs/smart_ptr/sp_techniques.html#pimpl
// Not Pimpl-ed are class Platform, Device, and Function.
// These are also passed by value.

#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <iosfwd>

#include <boost/shared_ptr.hpp>
#include "gpu_incl.h" // ideally, this goes into the .cc, but too much leakage
#include <cufft.h>

#include <GPUProc/gpu_wrapper.h> // GPUException

#if CUDA_VERSION < 4020
typedef int CUsharedconfig;
#endif

namespace LOFAR
{
  namespace Cobalt
  {
    class PerformanceCounter;
    namespace gpu
    {

      // Exception class for CUDA errors.
      EXCEPTION_CLASS(CUDAException, GPUException);

      // Return the cuFFT error string associated with \a errcode.
      std::string cufftErrorMessage(cufftResult errcode);

      // Return the CUDA error string associated with \a errcode.
      std::string errorMessage(CUresult errcode);


      // Struct representing a CUDA Block, which is similar to the @c dim3 type
      // in the CUDA Runtime API.
      struct Block
      {
        Block(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1);
        unsigned int x;
        unsigned int y;
        unsigned int z;
        friend std::ostream& operator<<(std::ostream& os, const Block& block);
      };


      // Struct representing a CUDA Grid, which is similar to the @c dim3 type
      // in the CUDA Runtime API.
      struct Grid
      {
        Grid(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1);
        unsigned int x;
        unsigned int y;
        unsigned int z;
        friend std::ostream& operator<<(std::ostream& os, const Grid& grid);
      };

      // Forward declaration needed by Platform::devices.
      class Device;

      // This class is not strictly needed, because in CUDA there's only one
      // platform, but it hides the CUDA calls and makes it similar to OpenCL.
      class Platform
      {
      public:
        // Initialize the CUDA platform.
        // \param flags must be 0 (at least up till CUDA 5.0).
        Platform(unsigned int flags = 0);

        // The CUDA version (f.e. 5.0 -> 5000).
        int version() const;

        // Returns the number of devices in the CUDA platform.
        size_t size() const;

        // Returns a vector of all devices in the CUDA platform.
        std::vector<Device> devices() const;

        // Returns the name of the CUDA platform. (currently, "NVIDIA CUDA")
        std::string getName() const;

        // Return the maximum number of threads per block, that
        // is supported by all devices on the platform.
        // 
        // Hardware dependent.
        // - Returns at least 512 (except for ancient hardware)
        // - Returns 1024 for K10 (= Cobalt hardware)
        unsigned getMaxThreadsPerBlock() const;
      };

      // Wrap a CUDA Device.
      class Device
      {
      public:
        // Create a device.
        // \param ordinal is the device number; 
        //        valid range: [0, Platform.size()-1]
        Device(int ordinal = 0);

        // Order Devices by PCI ID (used in std::sort)
        bool operator<(const Device &other) const;

        // Return the name of the device in human readable form.
        std::string getName() const;

        // Return the compute capability (major)
        unsigned getComputeCapabilityMajor() const;

        // Return the compute capability (minor)
        unsigned getComputeCapabilityMinor() const;

        // Return the total amount of global memory, in bytes
        size_t getTotalGlobalMem() const;

        // Return the maximum amount of shared memory per block
        size_t getBlockSharedMem() const;

        // Return the total amount of constant memory
        size_t getTotalConstMem() const;

        // Return the PCI ID (bus:device) of this GPU
        std::string pciId() const;

        // Return the maximum number of threads per block
        // 
        // Hardware dependent.
        // - Returns at least 512 (except for ancient hardware)
        // - Returns 1024 for K10 (= Cobalt hardware)
        unsigned getMaxThreadsPerBlock() const;

        // Return the maximum dimensions of a block of threads.
        struct Block getMaxBlockDims() const;

        // Return the maximum dimensions of a grid of blocks.
        struct Grid getMaxGridDims() const;

        // Return the number of multi-processors.
        unsigned getMultiProcessorCount() const;

        // Return the maximum number of threads that can be
        // resident on a multi-processor.
        unsigned getMaxThreadsPerMultiProcessor() const;

        // Return information on a specific \a attribute.
        // \param attribute CUDA device attribute
        int getAttribute(CUdevice_attribute attribute) const;

      private:
        // Context needs access to our \c _device to create a context.
        friend class Context;

        // The CUDA device.
        CUdevice _device;
      };


      // Wrap a CUDA Context. Since this class manages a resource (a CUDA
      // context), it uses the pimpl idiom in combination with a reference
      // counted pointer to make it copyable.
      //
      // We do not tie any context to any thread by default -- all contexts
      // are `floating', and are to be tied to a thread only by pushing them
      // as the current context, performing operation(s), and popping them
      // from the current context stack. The pushing and popping is automated
      // in the ScopedCurrentContext class.
      class Context
      {
      public:
        // Create a new CUDA context and associate it with the calling thread.
        // In other words, \c setCurrent() is implied.
        Context(const Device &device, unsigned int flags = CU_CTX_SCHED_AUTO);

        // Returns the device associated to the _current_ context.
        Device getDevice() const;

        // Set the cache configuration of the _current_ context.
        void setCacheConfig(CUfunc_cache config) const;

        // Set the shared memory configuration of the _current_ context.
        void setSharedMemConfig(CUsharedconfig config) const;

      private:
        // Non-copyable implementation class.
        class Impl;

        // Reference counted pointer to the implementation class.
        boost::shared_ptr<Impl> _impl;

        friend class ScopedCurrentContext;
      };


      // Make a certain context the current one for a certain scope.
      class ScopedCurrentContext
      {
      public:
        ScopedCurrentContext( const Context &context );
        ~ScopedCurrentContext();

      private:
        const Context &_context;
      };


      // Wrap CUDA Host Memory. This is the equivalent of a OpenCL Buffer. CUDA
      // distinguishes between between host- and device memory, OpenCL does not.
      class HostMemory
      {
      public:
        // Allocate \a size bytes of host memory.
        // \param context CUDA context associated with this HostMemory object.
        // \param size number of bytes to allocate
        // \param flags affect allocation
        // \note To create pinned memory, we need to set
        // \code
        // flags = CU_MEMHOSTALLOC_PORTABLE
        // \endcode
        // \note For input buffers we may consider setting
        // \code
        // flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_WRITECOMBINED
        // \endcode
        // Please refer to the documentation of the function \c cuMemHostAlloc()
        // in the CUDA Driver API for details.
        HostMemory(const Context &context, size_t size, unsigned int flags = 0);

        // Return a pointer to the actual memory.
        // \warning The returned pointer shall not have a lifetime beyond the
        // lifetime of this object (actually the last copy).
        template <typename T>
        T *get() const;

        // Return the size of this memory block.
        size_t size() const;

      private:
        // Get a void pointer to the actual memory from our Impl class. This
        // method is only used by our templated get() method.
        void* getPtr() const;

        // Non-copyable implementation class.
        class Impl;

        // Reference counted pointer to the implementation class.
        boost::shared_ptr<Impl> _impl;
      };


      // Wrap CUDA Device Memory. This is the equivalent of an OpenCL
      // Buffer. CUDA distinguishes between between host- and device memory,
      // OpenCL does not.
      class DeviceMemory
      {
      public:
        // Allocate \a size bytes of device memory.
        DeviceMemory(const Context &context, size_t size);

        // Return a device pointer as a handle to the memory.
        void *get() const;

        // Fill the first \a n bytes of memory with the constant byte \a uc.
        // \param uc Constant byte value to put into memory
        // \param n  Number of bytes to set. Defaults to the complete block.
        //           If \a n is larger than the current memory block size, then
        //           the complete block will be set to \a uc.
        void set(unsigned char uc, size_t n = (size_t)-1);

        // Return the size of this memory block.
        size_t size() const;

        // Fetch the contents of this buffer in a new HostMemory buffer.
        HostMemory fetch() const;

      private:
        // Function needs access to our device ptr location to set this as a kernel arg.
        friend class Function;

        // Non-copyable implementation class.
        class Impl;

        // Reference counted pointer to the implementation class.
        boost::shared_ptr<Impl> _impl;
      };


      // Wrap a CUDA Module. This is the equivalent of a OpenCL Program.
      class Module
      {
      public:
        typedef std::map<CUjit_option, void*> optionmap_t;

        Module(); // TODO: tmp, as long as CorrelatorPipelinePrograms needs a default init

        // Load the module in the file \a fname into the given \a context. The
        // file should be a \e cubin file or a \e ptx file as output by \c nvcc.
        // \param context CUDA context associated with this Module object.
        // \param fname name of a module file
        // \note For details, please refer to the documentation of \c
        // cuModuleLoad in the CUDA Driver API.
        Module(const Context &context, const std::string &fname);

        // Load the module pointed to by \a image into the given \a context. The
        // pointer may point to a null-terminated string containing \e cubin or
        // \e ptx code.
        // \param context CUDA context associated with this Module object.
        // \param image pointer to a module image in memory
        // \note For details, please refer to the documentation of \c
        // cuModuleLoadData in the CUDA Driver API.
        Module(const Context &context, const void *image);

        // Load the module pointed to by \a image into the given \a context. The
        // pointer may point to a null-terminated string containing \e cubin or
        // \e ptx code.
        // \param context CUDA context associated with this Module object.
        // \param image pointer to a module image in memory
        // \param options map of \c CUjit_option items, with their associated
        // values.
        // \note All values are cast to void*, so if an option requires
        // an unsigned int as value, the unsigned int's value itself is cast to void*!
        // \note For details, please refer to the documentation of \c
        // cuModuleLoadDataEx in the CUDA Driver API.
        Module(const Context &context, const void *image, optionmap_t &options);

        // Return the Context in which this Module was created.
        Context getContext() const;

      private:
        // Function needs access to our module to create a function.
        friend class Function;

        // Non-copyable implementation class.
        class Impl;

        // Reference counted pointer to the implementation class.
        boost::shared_ptr<Impl> _impl;
      };

      // Wrap a CUDA Device Function. This is the equivalent of an OpenCL
      // Program.
      class Function
      {
      public:
        // Construct a function object by looking up the function \a name in the
        // module \a module.
        Function(const Module &module, const std::string &name);

        // Return the name of the function.
        std::string name() const;

        // Set kernel immediate argument number \a index to \a val.
        // \a val must outlive kernel execution.
        // Not for device memory objects (be it as DeviceMemory or as void *).
        template <typename T>
        void setArg(size_t index, const T &val);

        // Set kernel DeviceMemory object argument number \a index to \a mem.
        // \a mem must outlive kernel execution.
        void setArg(size_t index, const DeviceMemory &mem);

        // Set pointer to kernel device memory object (as void *) number \a index
        // to \a val. \a *val must outlive kernel execution.
        // Note: Prefer to use setArg() passing a DeviceMemory ref over this overload.
        void setArg(size_t index, const void **val);

        // Return information about a function.
        // \note For details on valid values for \a attribute, please refer to
        // the documentation of cuFuncGetAttribute in the CUDA Driver API.
        int getAttribute(CUfunction_attribute attribute) const;

        // Set the shared memory configuration for a device function.
        // \note For details on valid values for \a config, please refer to the
        // documentation of cuFuncSetSharedMemConfig in the CUDA Driver API.
        void setSharedMemConfig(CUsharedconfig config) const;

      private:
        const Context _context;

        // Keep the Module alive, because Function actually wraps a pointer
        // to a function within the Module.
        const Module _module;

        // The name of the function, for error reporting purposes
        const std::string _name;

        // Stream needs access to our CUDA function to launch a kernel.
        friend class Stream;

        // CUDA function.
        CUfunction _function;

        // Function arguments as set.
        std::vector<const void *> _kernelArgs;

        // Helper function to modify _kernelArgs.
        void doSetArg(size_t index, const void *argp);

        // Do not use. To guard against passing pointers.
        // Note that even device void * cannot be passed, because we need its
        // address with a life time longer than this formal parameter.
        template<typename T>
        void setArg(size_t index, const T *&); // intentionally not impl.

        // Do not use. To guard against passing HostMemory references to kernels.
        void setArg(size_t index, const HostMemory &); // intentionally not impl.

        // Do not use. To guard against passing HostMemory pointers to kernels.
        void setArg(size_t index, const HostMemory *); // intentionally not impl.
      };

      // Wrap a CUDA Event. This is the equivalent of an OpenCL Event.
      class Event
      {
      public:
        // Construct a CUDA event. This class manages a resource (a CUDA event)
        // and is therefore implemented using the pimpl idiom, using a reference
        // counted pointer to a non-copyable implementation class.
        // \note For details on valid values for \a flags, please refer to the
        // documentation of cuEventCreate in the CUDA Driver API.
        Event(const Context &context, unsigned int flags = CU_EVENT_DEFAULT);

        // Return the elapsed time in milliseconds between this event and the \a
        // second event.
        float elapsedTime(Event &second) const;

        // Wait until all work preceding this event in the same stream has
        // completed.
        void wait();

      private:
        // Stream needs access to our CUDA event to wait for and record events.
        friend class Stream;

        // Non-copyable implementation class.
        class Impl;

        // Reference counted pointer to the implementation class.
        boost::shared_ptr<Impl> _impl;
      };


      // Wrap a CUDA Stream. This is the equivalent of an OpenCL
      // CommandQueue. This class manages a resource (a CUDA stream) and is
      // therefore implemented using the pimpl idiom, using a reference counted
      // pointer to a non-copyable implementation class.
      class Stream
      {
      public:
        // Create a stream.
        // \param flags must be 0 for CUDA < 5.0
        // \param context CUDA context associated with this Stream object.
        // \note For details on valid values for \a flags, please refer to the
        // documentation of \c cuStreamCreate in the CUDA Driver API.
        explicit Stream(const Context &context, unsigned int flags = 0);  // named CU_STREAM_DEFAULT (0) since CUDA 5.0

        // Transfer data from host memory \a hostMem to device memory \a devMem.
        // \param devMem Device memory that will be copied to.
        // \param hostMem Host memory that will be copied from.
        // \param synchronous Indicates whether the transfer must be done
        //        synchronously or asynchronously.
        void writeBuffer(const DeviceMemory &devMem, const HostMemory &hostMem,
                         bool synchronous = false) const;

        // Transfer data from host memory \a hostMem to device memory \a devMem.
        // When gpuProfiling is enabled this transfer is synchronous
        // \param devMem Device memory that will be copied to.
        // \param hostMem Host memory that will be copied from.
        // \param counter PerformanceCounter that will receive transfer duration
        // if  gpuProfiling is enabled
        // \param synchronous Indicates whether the transfer must be done
        //        synchronously or asynchronously. Default == false
        void writeBuffer(const DeviceMemory &devMem, const HostMemory &hostMem,
                         const PerformanceCounter &counter, bool synchronous = false) const;

        // Transfer data from device memory \a devMem to host memory \a hostMem.
        // \param hostMem Host memory that will be copied to.
        // \param devMem Device memory that will be copied from.
        // \param synchronous Indicates whether the transfer must be done
        //        synchronously or asynchronously.
        void readBuffer(const HostMemory &hostMem, const DeviceMemory &devMem,
                        bool synchronous = false) const;

        // Transfer data from device memory \a devMem to host memory \a hostMem.
        // When gpuProfiling is enabled this transfer is synchronous
        // \param hostMem Host memory that will be copied to.
        // \param devMem Device memory that will be copied from.
        // \param counter PerformanceCounter that will receive transfer duration
        // if  gpuProfiling is enabled
        // \param synchronous Indicates whether the transfer must be done
        //        synchronously or asynchronously. Default == false
        void readBuffer(const HostMemory &hostMem, const DeviceMemory &devMem,
                        const PerformanceCounter &counter, bool synchronous = false) const;

        // Transfer data from device memory \a devSource to device memory \a devTarget.
        // \param devTarget Device memory that will be copied to.
        // \param devSource Device memory that will be copied from.
        // \param synchronous Indicates whether the transfer must be done
        //        synchronously or asynchronously.
        void copyBuffer(const DeviceMemory &devTarget, const DeviceMemory &devSource,
                        bool synchronous = false) const;

        // Transfer data from device memory \a devSource to device memory \a devTarget.
        // When gpuProfiling is enabled this transfer is synchronous
        // \param devTarget Device memory that will be copied to.
        // \param devSource Device memory that will be copied from.
        // \param counter PerformanceCounter that will receive transfer duration
        //        if gpuProfiling is enabled
        // \param synchronous Indicates whether the transfer must be done
        //        synchronously or asynchronously. Defaults to \c false
        //        (asynchronously).
        void copyBuffer(const DeviceMemory &devTarget, const DeviceMemory &devSource,
                        const PerformanceCounter &counter, bool synchronous = false) const;

        // Launch a CUDA function.
        // \param function object containing the function to launch
        // \param grid Grid size (in terms of blocks (not threads (OpenCL)))
        // \param block Block (thread group) size
        void launchKernel(const Function &function,
                          const Grid &grid, const Block &block) const;

        // Check if all operations on this stream have completed.
        // \return true if all completed, or false otherwise.
        bool query() const;

        // Wait until a this stream's tasks are completed.
        void synchronize() const;

        // Let this stream wait on the event \a event.
        void waitEvent(const Event &event) const;

        // Record the event \a event for this stream.
        void recordEvent(const Event &event) const;

        // Return the underlying CUDA stream. TODO: try to get rid of CUstream here: FFT thing to here or make it friend
        CUstream get() const;

        // Returns the context associated with the underlying CUDA stream.
        Context getContext() const; // TODO: consider using this in the SubbandProcs (now has Stream and Context stored)

        // Return whether this stream mandates synchronous behaviour
        bool isSynchronous() const;

      private:
        // Non-copyable implementation class.
        class Impl;

        // Reference counted pointer to the implementation class.
        boost::shared_ptr<Impl> _impl;

        // Force synchronous transfers and kernel launches
        bool force_synchronous;
      };

    } // namespace gpu
  } // namespace Cobalt
} // namespace LOFAR

#include "gpu_wrapper.tcc"

#endif


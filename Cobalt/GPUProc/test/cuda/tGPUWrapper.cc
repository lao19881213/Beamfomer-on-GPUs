//# tGPUWrapper.cc: test CUDA wrapper classes
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
//# $Id: tGPUWrapper.cc 27064 2013-10-23 13:37:36Z klijn $

#include <lofar_config.h>

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>

#include <Common/LofarLogger.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/cuda/PerformanceCounter.h>
#include <UnitTest++.h>

using namespace std;
using namespace LOFAR::Cobalt;
using namespace LOFAR::Cobalt::gpu;

SUITE(Platform) {
  // Test construction and destruction
  TEST(Basic) {
    Platform pf;

    cout << "CUDA version:  " << pf.version() << endl;
    cout << "Platform name: " << pf.getName() << endl;
    cout << "Nr of devices: " << pf.size() << endl;
  }
}

SUITE(Device) {
  TEST(ListDevices) {
    Platform pf;
    vector<Device> devices(pf.devices());

    size_t globalMaxThreadsPerBlock = pf.getMaxThreadsPerBlock();

    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      const Device &dev = *i;

      cout << "Device:         " << dev.getName() << endl;
      cout << " Capability:    " << dev.getComputeCapabilityMajor() << "." << dev.getComputeCapabilityMinor() << endl;
      cout << " Global Memory: " << setw(4) << dev.getTotalGlobalMem()/1024/1024 << " MByte" << endl;
      cout << " Shared Memory: " << setw(4) << dev.getBlockSharedMem()/1024 << " KByte/block" << endl;
      cout << " Const Memory:  " << setw(4) << dev.getTotalConstMem()/1024 << " KByte" << endl;
      cout << " Threads/block: " << setw(4) << dev.getMaxThreadsPerBlock() << endl;

      CHECK(dev.getMaxThreadsPerBlock() <= globalMaxThreadsPerBlock);
    }
  }
}

SUITE(Context) {
  TEST(Basic) {
    Platform pf;
    vector<Device> devices(pf.devices());

    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      Context ctx(*i);

      // Context should report the same device as it was created with
      CHECK_EQUAL(i->getName(), ctx.getDevice().getName());
    }
  }
}

SUITE(Stream) {
  TEST(Basic) {
    Platform pf;
    vector<Device> devices(pf.devices());

    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      Context ctx(*i);
      Stream s(ctx);

      // All operations should have been completed, as there were none
      CHECK_EQUAL(true, s.query());
    }
  }

  TEST(MultiContext) {
    Platform pf;
    vector<Device> devices(pf.devices());
    vector<Context> ctxs;
    vector<Stream> streams;

    // Create all contexts, 2 per device to ensure multiple contexts even if
    // we have one device.
    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      for (size_t n = 0; n < 2; ++n) {
        Context ctx(*i);
        ctxs.push_back(ctx);
      }
    }

    // Create all streams
    for (vector<Context>::const_iterator i = ctxs.begin(); i != ctxs.end(); ++i) {
      Stream s(*i);
      streams.push_back(s);
    }

    // Query all streams
    for (vector<Stream>::const_iterator i = streams.begin(); i != streams.end(); ++i) {
      // All operations should have been completed, as there were none
      CHECK_EQUAL(true, i->query());
    }
  }
}

SUITE(Memory) {
  TEST(HostMemoryAlloc) {
    Platform pf;
    vector<Device> devices(pf.devices());

    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      Context ctx(*i);

      // Allocate some HostMemory
      const size_t size = 1024 * 1024;
      HostMemory hm(ctx, size);

      CHECK(hm.get<void>() != NULL);
      CHECK_EQUAL(size, hm.size());

      // Fill the memory
      memset(hm.get<void>(), 0, size);
    }
  }

  TEST(DeviceMemoryAlloc) {
    Platform pf;
    vector<Device> devices(pf.devices());

    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      Context ctx(*i);

      // Allocate some DeviceMemory
      const size_t size = 1024 * 1024;
      DeviceMemory dm(ctx, size);

      CHECK(dm.get() != NULL);
      CHECK_EQUAL(size, dm.size());
    }
  }

  TEST(Transfer) {
    Platform pf;
    vector<Device> devices(pf.devices());

    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      Context ctx(*i);     
      Stream s(ctx);

      PerformanceCounter counter(ctx);
      // Allocate memory on host and device
      const size_t size = 1024 * 1024;
      HostMemory hm(ctx, size);
      DeviceMemory dm(ctx, size);

      // Fill the host memory
      memset(hm.get<void>(), 42, size);

      // Transfer to device
      s.writeBuffer(dm, hm, counter, true);

      // Clear host memory
      memset(hm.get<void>(), 0, size);

      // Transfer back
      s.readBuffer(hm, dm, true);

      // Check results
      for (size_t i = 0; i < size; ++i) {
        CHECK_EQUAL(42, hm.get<char>()[i]);
      }
    }
  }

  TEST(TransferDeviceDevice) {
    Platform pf;
    vector<Device> devices(pf.devices());


    for (vector<Device>::const_iterator i = devices.begin(); i != devices.end(); ++i) {
      Context ctx(*i);
      Stream s(ctx);
      PerformanceCounter counter(ctx);

      // Allocate memory on host and device
      const size_t size = 1024 * 1024;
      HostMemory hm(ctx, size);
      DeviceMemory dm(ctx, size);
      DeviceMemory dm_target(ctx, size);

      // Fill the host memory
      memset(hm.get<void>(), 42, size);

      // Transfer to device
      s.writeBuffer(dm, hm, true);

      //copy between buffers on the device
      s.copyBuffer(dm_target, dm, counter, true);

      // Clear host memory
      memset(hm.get<void>(), 0, size);

      // Transfer back
      s.readBuffer(hm, dm_target, true);

      // Check results
      for (size_t i = 0; i < size; ++i) {
        CHECK_EQUAL(42, hm.get<char>()[i]);
      }
    }
  }
}

int main(int, char **) {
  INIT_LOGGER("tGPUWrapper");
  try {
    Platform pf;
    return UnitTest::RunAllTests() > 0;
  } catch (GPUException& e) {
    cerr << "No GPU device(s) found. Skipping tests." << endl;
    return 0;
  }
}


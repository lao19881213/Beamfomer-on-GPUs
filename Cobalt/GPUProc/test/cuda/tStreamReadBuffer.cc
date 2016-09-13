//# tStreamReadBuffer.cc
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
//# $Id: tStreamReadBuffer.cc 25141 2013-05-31 13:18:25Z loose $

#include <lofar_config.h>

#include <cstddef>
#include <cstring>
#include <iostream>

#include <GPUProc/gpu_wrapper.h>
#include <Common/LofarLogger.h>

using namespace std;
using namespace LOFAR::Cobalt;

int main() {
  INIT_LOGGER("tStreamReadBuffer");
  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    return 3;
  }

  gpu::Device dev(0);
  gpu::Context ctx(dev);

  const size_t bufSize = 16 * 1024 * 1024;
  const char expectedVal = 42;
  gpu::HostMemory hBuf1(ctx, bufSize);
  gpu::HostMemory hBuf2(ctx, bufSize);
  char *buf1 = hBuf1.get<char>();
  char *buf2 = hBuf2.get<char>();
  memset(buf1, expectedVal, bufSize * sizeof(char));
  memset(buf2,           0, bufSize * sizeof(char));

  gpu::DeviceMemory dBuf(ctx, bufSize);

  gpu::Stream strm(ctx);

  strm.writeBuffer(dBuf, hBuf1, false);
  strm.synchronize();

  // implicitly synchronous read back
  strm.readBuffer(hBuf2, dBuf, true);

  // check if data is there
  size_t nrUnexpectedVals = 0;
  for (size_t i = bufSize; i > 0; ) {
    i--;
    if (buf2[i] != expectedVal) {
      nrUnexpectedVals += 1;
    }
  }

  if (nrUnexpectedVals > 0) {
    cerr << "Got > 0 unexpected values: " << nrUnexpectedVals << endl;
    return 1;
  }

  return 0;
}


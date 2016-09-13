//# tMultiDimArrayHostBuffer.cc: test program for the MultiDimArrayHostBuffer
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
//# $Id: tMultiDimArrayHostBuffer.cc 25246 2013-06-10 12:12:36Z amesfoort $

#include <lofar_config.h>

#include <cstring>

#include <Common/LofarLogger.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>

using namespace std;
using namespace LOFAR::Cobalt;

static bool test1(gpu::Context &ctx, gpu::Stream &queue, size_t dim1)
{
  MultiDimArrayHostBuffer<float, 1> b1(boost::extents[dim1], ctx);
  ASSERT(b1.size() == dim1 * sizeof(float));

  MultiDimArrayHostBuffer<float, 1> b1out(boost::extents[dim1], ctx);

  // initialize b1
  for (size_t i = 0; i < dim1; i++)
  {
    b1[i] = i;
  }

  gpu::DeviceMemory d1(ctx, dim1 * sizeof(float));

  queue.writeBuffer(d1, b1, false);
  queue.readBuffer(b1out, d1, true);

  // check
  return memcmp(b1.get<void>(), b1out.get<void>(), dim1 * sizeof(float)) == 0;
}

static bool test2(gpu::Context &ctx, gpu::Stream &queue, size_t dim1, size_t dim2)
{
  MultiDimArrayHostBuffer<float, 2> b2(boost::extents[dim1][dim2], ctx);
  ASSERT(b2.size() == dim1 * dim2 * sizeof(float));

  MultiDimArrayHostBuffer<float, 2> b2out(boost::extents[dim1][dim2], ctx);

  // initialize b2
  for (size_t i = 0; i < dim1; i++)
    for (size_t j = 0; j < dim2; j++)
  {
    b2[i][j] = i * dim2 + j;
  }

  gpu::DeviceMemory d2(ctx, dim1 * dim2 * sizeof(float));

  queue.writeBuffer(d2, b2, false);
  queue.readBuffer(b2out, d2, true);

  // check
  return memcmp(b2.get<void>(), b2out.get<void>(), dim1 * dim2 * sizeof(float)) == 0;
}

static bool test3(gpu::Context &ctx, gpu::Stream &queue, size_t dim1, size_t dim2, size_t dim3)
{
  MultiDimArrayHostBuffer<float, 3> b3(boost::extents[dim1][dim2][dim3], ctx);
  ASSERT(b3.size() == dim1 * dim2 * dim3 * sizeof(float));

  MultiDimArrayHostBuffer<float, 3> b3out(boost::extents[dim1][dim2][dim3], ctx);

  // initialize b2
  for (size_t i = 0; i < dim1; i++)
    for (size_t j = 0; j < dim2; j++)
      for (size_t k = 0; k < dim3; k++)
  {
    b3[i][j][k] = i * dim2 * dim3 + j * dim3 + k;
  }

  gpu::DeviceMemory d3(ctx, dim1 * dim2 * dim3 * sizeof(float));

  queue.writeBuffer(d3, b3, false);
  queue.readBuffer(b3out, d3, true);

  // check
  return memcmp(b3.get<void>(), b3out.get<void>(), dim1 * dim2 * dim3 * sizeof(float)) == 0;
}

int main() {
  INIT_LOGGER("tMultiDimArrayHostBuffer");

  // Set up gpu environment
  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " CUDA devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    return 3;
  }
  gpu::Device device(0);
  gpu::Context ctx(device);
  gpu::Stream queue(ctx);

  const size_t dims[] = {1 * 1024, 2 * 1024, 2};

  bool ok1 = test1(ctx, queue, dims[0]);
  if (!ok1) LOG_ERROR("test 1 failed");
  bool ok2 = test2(ctx, queue, dims[0], dims[1]);
  if (!ok2) LOG_ERROR("test 2 failed");
  bool ok3 = test3(ctx, queue, dims[0], dims[1], dims[2]);
  if (!ok3) LOG_ERROR("test 3 failed");

  return ok1 && ok2 && ok3 ? 0 : 1;
}


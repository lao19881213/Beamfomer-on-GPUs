//# tKernel.in_.cu: simple function to test Kernel class
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
//# $Id: tKernel.in_.cu 24903 2013-05-14 23:50:58Z amesfoort $

extern "C" {

  // test various "types" of args (for arg setting), esp. an immediate and a buffer
  __global__ void testKernel(float *out, const float *in, size_t size, float inc)
  {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
      out[i] = in[i] + inc;
    }
  }

}


//# cmpfloat.cc
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
//# $Id: cmpfloat.cc 26258 2013-08-28 15:43:02Z amesfoort $

//#include <lofar_config.h>

#include <cstdlib>
#include <iostream>
#include <fstream>

#include "fpequals.h"

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;

using LOFAR::Cobalt::fpEquals;


int main(int argc, char *argv[])
{
  cerr.precision(8); // print full float precision (7 + the 0.).

  // Default epsilon. For fp cmp, you want to override this for sure.
  double epsilon = std::numeric_limits<double>::epsilon();

  if (argc < 3 || argc > 4)
  {
    cerr << "Usage: " << argv[0] << "[1.0e-12] <file1> <file2>" << endl;
    cerr << "  where the optional floating point argument overrides the comparison epsilon" << endl;
    return 1;
  }

  char *filename1;
  char *filename2;

  if (argc == 3)
  {
    filename1 = argv[1];
    filename2 = argv[2];
  } else { // argc == 4
    filename1 = argv[2];
    filename2 = argv[3];

    epsilon = std::atof(argv[1]);
    if (epsilon <= 0.0 || epsilon > 1.0)
    {
      cerr << "Epsilon command line argument is out of range" << endl;
      return 1;
    } else {
      cout << "Using an epsilon of " << epsilon << endl;
    }
  }
  float eps = (float)epsilon; // atm, we only cmp single precision floats

  ifstream ifs1(filename1, std::ios::binary);
  if (!ifs1)
  {
    cerr << "Failed to open file " << filename1 << endl;
    return 1;
  }

  ifstream ifs2(filename2, std::ios::binary);
  if (!ifs2)
  {
    cerr << "Failed to open file " << filename2 << endl;
    return 1;
  }

  const size_t bufLen = 2048;
  float *buf1 = new float[bufLen];
  float *buf2 = new float[bufLen];

  int status = 0;
  size_t total = 0;

  while (ifs1.good() && ifs2.good()) {
    size_t len = bufLen;
    size_t nbytes1, nbytes2;

    ifs1.read(reinterpret_cast<char *>(buf1), bufLen);
    nbytes1 = ifs1.gcount();

    ifs2.read(reinterpret_cast<char *>(buf2), bufLen);
    nbytes2 = ifs2.gcount();

    if (nbytes1 == 0 || nbytes1 != nbytes2 ||
        nbytes1 % sizeof(float) != 0 || nbytes2 % sizeof(float) != 0)
    {
      cerr << "Failed to read an equal amount of bytes of at least a float from both input streams" << endl;
      status = 1;
      break;
    }

    if (nbytes1 < len * sizeof(float))
      len = nbytes1 / sizeof(float);

    for (size_t i = 0; i < len; i++)
    {
      if (!fpEquals(buf1[i], buf2[i], eps))
      {
        cerr << "Error: value diff beyond eps at pos " << total + i << ": " << buf1[i] << " " << buf2[i] << endl;
        status = 1;
      }
    }

    total += len;
  }

  if (!ifs1.eof())
  {
    cerr << "Error occurred while reading from file " << filename1 << endl;
    status = 1;
  }

  if (!ifs2.eof())
  {
    cerr << "Error occurred while reading from file " << filename2 << endl;
    status = 1;
  }

  delete[] buf2;
  delete[] buf1;

  return status;
}


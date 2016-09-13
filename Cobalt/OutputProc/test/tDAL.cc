//# tDAL.cc: Test HDF5 routines through DAL
//# Copyright (C) 2011-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: tDAL.cc 24462 2013-03-28 14:44:39Z mol $

#include <lofar_config.h>

#include <string>
#include <iostream>

#ifdef HAVE_DAL
#include <dal/dal_version.h>

using namespace dal;
#endif

using namespace std;

int main()
{
#ifdef HAVE_DAL
  if (!check_hdf5_versions()) {
    cerr << "HDF5 version mismatch. DAL was compiled with " << version_hdf5_headers_dal() << ", our headers are " << version_hdf5_headers_current() << ", our library is " << version_hdf5() << endl;
    return 1;
  }
#else
  cout << "Built without DAL, skipped actual test code." << endl;
#endif

  return 0;
}


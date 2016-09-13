//# tMSWriterDAL: Test HDF5 routines through DAL
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
//# $Id: tMSWriterDAL.cc 24388 2013-03-26 11:14:29Z amesfoort $

#include <lofar_config.h>

#include <CoInterface/DataFactory.h>
#include <CoInterface/StreamableData.h>
#include <CoInterface/Parset.h>

#ifdef HAVE_DAL
#include <OutputProc/MSWriterDAL.h>
#endif

using namespace std;
using namespace LOFAR;
using namespace Cobalt;

#if defined WORDS_BIGENDIAN
const int bigEndian = 1;
#else
const int bigEndian = 0;
#endif

int main()
{
#ifdef HAVE_DAL
  Parset parset("tMSWriterDAL.parset");

  {
    MSWriterDAL<float,3> writer("tMSWriterDAL_tmp.h5", parset, 0, bigEndian);

    StreamableData *data = newStreamableData(parset, BEAM_FORMED_DATA, 0);

    writer.write(data);

    delete data;
  }
#else
  cout << "Built without DAL, skipped actual test code." << endl;
#endif

  return 0;
}


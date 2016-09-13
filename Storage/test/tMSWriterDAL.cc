//# tMSWriterDAL: Test HDF5 routines through DAL
//#
//#  Copyright (C) 2011
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: $

#include <lofar_config.h>

#ifdef HAVE_DAL

#include <Storage/MSWriterDAL.h>
#include <Interface/DataFactory.h>

using namespace std;
using namespace LOFAR;
using namespace RTCP;

#if defined WORDS_BIGENDIAN
const int bigEndian = 1;
#else
const int bigEndian = 0;
#endif

int main() {
  Parset parset("tMSWriterDAL.parset");

  {
    MSWriterDAL<float,3> writer("tMSWriterDAL_tmp.h5", parset, 0, bigEndian);

    StreamableData *data = newStreamableData(parset, BEAM_FORMED_DATA, 0);

    writer.write(data);

    delete data;
  }  

  return 0;
}

#else

int main() {
  return 0;
}
#endif

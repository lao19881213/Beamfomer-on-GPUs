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

#include <Common/LofarLogger.h>
#include <Storage/MSWriterCorrelated.h>
#include <Interface/DataFactory.h>
#include <Interface/FinalMetaData.h>

using namespace std;
using namespace LOFAR;
using namespace RTCP;

#if defined WORDS_BIGENDIAN
const int bigEndian = 1;
#else
const int bigEndian = 0;
#endif

int main() {
  INIT_LOGGER("tMSWriterCorrelated");

  Parset parset("tMSWriterCorrelated.parset");

  {
    // Create MeasurementSet
    MSWriterCorrelated writer("", "tMSWriterCorrelated.in_1/SB000.MS", parset, 0, bigEndian);

    // Write some data
    StreamableData *data = newStreamableData(parset, CORRELATED_DATA, 0);

    writer.write(data);

    delete data;

    // Add broken tile information
    FinalMetaData fmd;
    struct FinalMetaData::BrokenRCU rcu;

    rcu.station = "CS013";
    rcu.time = "2012-01-01 00:00:00";
    rcu.type = "RCU";
    rcu.seqnr = 2;

    fmd.brokenRCUsAtBegin.push_back(rcu);

    rcu.station = "CS013";
    rcu.time = "2012-01-01 00:00:00";
    rcu.type = "LBA";
    rcu.seqnr = 4;

    fmd.brokenRCUsAtBegin.push_back(rcu);

    writer.augment(fmd);
  }  

  return 0;
}

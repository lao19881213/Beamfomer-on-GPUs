//# DummyStorage.cc
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
//# $Id: DummyStorage.cc 25646 2013-07-12 09:58:39Z mol $

#include <lofar_config.h>

#include <string>
#include <iostream>
#include <boost/lexical_cast.hpp>

#include <Common/LofarLogger.h>
#include <Common/Thread/Mutex.h>
#include <Stream/PortBroker.h>
#include <CoInterface/Stream.h>
#include <CoInterface/FinalMetaData.h>
#include <CoInterface/Parset.h>

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

int observationID;
unsigned rank;

FinalMetaData origFinalMetaData;

Mutex logMutex;

void emulateStorage()
{
  // establish control connection
  string resource = getStorageControlDescription(observationID, rank);
  PortBroker::ServerStream stream(resource);

  // read and print parset
  Parset parset(&stream);
  {
    ScopedLock sl(logMutex);
    cout << "Storage: Parset received." << endl;
  }

  // read and print meta data
  FinalMetaData finalMetaData;
  finalMetaData.read(stream);
  {
    ScopedLock sl(logMutex);

    ASSERT(finalMetaData.brokenRCUsAtBegin == origFinalMetaData.brokenRCUsAtBegin);
    ASSERT(finalMetaData.brokenRCUsDuring  == origFinalMetaData.brokenRCUsDuring);

    cout << "Storage: FinalMetaData received and matches." << endl;
  }

  // write LTA feedback
  Parset feedbackLTA;
  feedbackLTA.add("foo", "bar");
  feedbackLTA.write(&stream);
}

void emulateFinalMetaDataGatherer()
{
  // establish control connection
  string resource = getStorageControlDescription(observationID, -1);
  PortBroker::ServerStream stream(resource);

  // read and print parset
  Parset parset(&stream);
  {
    ScopedLock sl(logMutex);
    cout << "FinalMetaDataGatherer: Parset received." << endl;
  }

  // set and write meta data
  origFinalMetaData.brokenRCUsAtBegin.push_back( FinalMetaData::BrokenRCU("CS001", "LBA", 2, "2012-01-01 12:34") );
  origFinalMetaData.brokenRCUsAtBegin.push_back( FinalMetaData::BrokenRCU("RS205", "HBA", 1, "2012-01-01 12:34") );
  origFinalMetaData.brokenRCUsDuring.push_back( FinalMetaData::BrokenRCU("DE601", "RCU", 3, "2012-01-01 12:34") );
  origFinalMetaData.write(stream);

  {
    ScopedLock sl(logMutex);
    cout << "FinalMetaDataGatherer: FinalMetaData sent." << endl;
  }
}

int main(int argc, char **argv)
{
  INIT_LOGGER("DummyStorage");

  ASSERT(argc == 4);

  observationID = boost::lexical_cast<int>(argv[1]);
  rank = boost::lexical_cast<unsigned>(argv[2]);
  //bool isBigEndian = boost::lexical_cast<bool>(argv[3]);

  // set up broker server
  PortBroker::createInstance(storageBrokerPort(observationID));

#pragma omp parallel sections
  {
#   pragma omp section
    try {
      emulateStorage();
    } catch (Exception &ex) {
      cout << "Storage caught exception: " << ex << endl;
    }

#   pragma omp section
    try {
      emulateFinalMetaDataGatherer();
    } catch (Exception &ex) {
      cout << "FinalMetaDataGatherer caught exception: " << ex << endl;
    }
  }
}


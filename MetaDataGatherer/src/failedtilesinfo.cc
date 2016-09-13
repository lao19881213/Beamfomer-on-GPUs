//# failedtilesinfo.cc: extract failed tiles info from the SAS database
//# Copyright (C) 2011
//# ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O.Box 2, 7990 AA Dwingeloo, The Netherlands
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
//# $Id: failedtilesinfo.cc 18832 2011-09-19 17:22:32Z duscha $
//#
//# @author Sven Duscha

#include <lofar_config.h>

// LOFAR
#include <Common/ParameterSet.h>
#include <Common/LofarLogger.h>
#include <Common/SystemUtil.h>    // needed for basename
#include <Common/StringUtil.h>    // needed for split
#include <Common/Exception.h>     // THROW macro for exceptions

// SAS
#include <OTDB/OTDBconstants.h>
#include <OTDB/OTDBconnection.h>
#include <OTDB/OTDBnode.h>
#include <OTDB/TreeMaintenance.h>
#include <OTDB/TreeValue.h>
#include <OTDB/ClassifConv.h>
#include <OTDB/Converter.h>
#include <OTDB/TreeTypeConv.h>

// STL
#include <iostream>

// Casacore
#include <measures/Measures/MEpoch.h>
#include <casa/Quanta/MVTime.h>

// Boost
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace LOFAR;
using namespace LOFAR::OTDB;
using namespace casa;
using namespace boost::posix_time;

// Use a terminate handler that can produce a backtrace.
Exception::TerminateHandler t(Exception::terminate);

// Time converter helper functions
MVEpoch toCasaTime (const string& time);
string fromCasaTime (const MVEpoch& epoch);

// Get the failed tile info before the given date (if end=0)
// or between the dates.
void getFailedTilesInfo(OTDBconnection& conn, 
                        const string& filename,
                        const string& timeStart,
                        const string& timeEnd=string());


int main (int argc, char* argv[])
{
  // Init logger
  string progName = LOFAR::basename(argv[0]);
  INIT_LOGGER(progName);
  try {
    // Get the parset name; use default if not given as first argument.
    string parsetName = "failedtilesinfo.parset";
    if (argc > 1) {
      parsetName = argv[1];
    }
    ParameterSet parset(parsetName);
    // Get the parameters.
    ///string host        = parset.getString("host", "sas.control.lofar.eu");
    string host        = parset.getString("host", "RS005.astron.nl");
    string db          = parset.getString("db", "TESTLOFAR_4");
    string user        = parset.getString("user", "paulus");
    string password    = parset.getString("password", "boskabouter");
    string port        = parset.getString("port", "5432");
    // Files to save SAS hardware strings of broken and failed tiles.
    // Failed means tiles failing during the observations.
    // Broken means tiles already broken at the start of the observation.
    string failedfilename  = parset.getString("FailedTilesFile",
                                              "failedTiles.txt");
    string brokenfilename  = parset.getString("BrokenTilesFile",
                                              "brokenTiles.txt");
    string startTimeString = parset.getString("StartTime", "");
    string endTimeString   = parset.getString("EndTime", "");
    ASSERT (!(failedfilename.empty() || brokenfilename.empty()));
    ASSERT (!(startTimeString.empty() || endTimeString.empty()));
    MVEpoch startTime = toCasaTime(startTimeString);
    MVEpoch endTime   = toCasaTime(endTimeString);
    if (startTime.get() > endTime.get()) {
      THROW(Exception, "starttime " << startTimeString
            << " must be <= end time " << endTimeString);    
    }
    // Convert to time format (ISO) that Boost understands.
    startTimeString = fromCasaTime(startTime);
    endTimeString   = fromCasaTime(endTime);

    LOG_DEBUG_STR("Getting SAS antenna health information");
    OTDBconnection conn(user, password, db, host, port); 
    LOG_DEBUG("Trying to connect to the database");
    ASSERTSTR(conn.connect(), "Connnection failed");
    LOG_DEBUG_STR("Connection succesful: " << conn);
    // Get broken hardware strings from SAS
    getFailedTilesInfo (conn, brokenfilename, startTimeString);
    getFailedTilesInfo (conn, failedfilename, startTimeString, endTimeString);
  
  } catch (Exception& x) {
    LOG_FATAL_STR("Unexpected exception: " << x);
    return 1;
  }
  return 0;
}

// Convert a casacore time string time YYYY-Mon-DD TT:MM:SS.ss to an MVEpoch
MVEpoch toCasaTime (const string& time)
{
  // e.g. 2011-Mar-19 21:17:06.514000
  Quantity result;
  ASSERT(!time.empty());
  MVTime::read(result, time);
  return result;
}

string fromCasaTime (const MVEpoch& epoch)
{
  MVTime t (epoch.get());
  return t.getTime().ISODate();
}


// Get information about broken tiles from SAS database and store it in 
// an ASCII text file
void getFailedTilesInfo (OTDBconnection& conn, 
                         const string& filename,
                         const string& timeStart,
                         const string& timeEnd)
{
  ASSERT(!filename.empty());
  // Get OTDB info.
  TreeTypeConv TTconv(&conn);     // TreeType converter object
  ClassifConv CTconv(&conn);      // converter
  vector<OTDBvalue> valueList;    // OTDB value list
  vector<OTDBtree> treeList = conn.getTreeList(TTconv.get("hardware"),
                                               CTconv.get("operational"));
  ASSERTSTR(treeList.size(), "No hardware tree found, run tPICtree first");  
  treeIDType treeID = treeList[treeList.size()-1].treeID();
  LOG_DEBUG_STR ("Using tree " << treeID);
  OTDBtree treeInfo = conn.getTreeInfo(treeID);
  LOG_DEBUG_STR(treeInfo);
  LOG_DEBUG("Constructing a TreeValue object");
  TreeValue tv(&conn, treeID);
  // Create the output file.
  fstream outfile;
  outfile.open(filename.c_str(), ios::out);

  // Get list of all broken hardware from SAS for timestamp
  LOG_DEBUG("Searching for a Hardware tree");

  if (timeEnd.empty()) {
    // Getting tiles broken at beginning.
    valueList = tv.getBrokenHardware (time_from_string(timeStart));
  } else {
    // Getting tiles failed during observation.
    LOG_INFO_STR ("Getting failed hardware from "
                  << timeStart << " to " << timeEnd);
    valueList = tv.getBrokenHardware (time_from_string(timeStart), 
                                      time_from_string(timeEnd));
  }

  if (valueList.empty()) {
    LOG_INFO_STR ("No failed hardware found.");
  } else {
    // Write entry in valuelist with broken hardware to file.
    // A broken antenna element/tile entry must contain .status_state
    int nrtile = 0;
    int nrrcu  = 0;
    for (unsigned int i=0; i<valueList.size(); i++) {
      if (valueList[i].name.find(".status_state") != string::npos) {
        vector<string> parts = StringUtil::split (valueList[i].name, '.');
        bool match = false;
        if (parts.size() > 4  &&  parts[4].size() > 3) {
          // parts[3] is station name; parts[4] is tile name/number
          string type = parts[4].substr(0,3);
          if (type == "LBA"  ||  type == "HBA") {
            outfile << parts[3] + ' ' + parts[4] << ' '
                    << valueList[i].time << endl;
            nrtile++;
            match = true;
          }
        }
        if (!match  &&  parts.size() > 7  &&  parts[7].size() > 3) {
          // parts[3] is station name; parts[7] is RCU name/number
          string type = parts[7].substr(0,3);
          if (type == "RCU") {
            outfile << parts[3] + ' ' + parts[7] << ' '
                    << valueList[i].time << endl;
            nrrcu++;
          }
        }
      }
    }
    LOG_INFO_STR ("Found " << nrtile << " broken tiles and "
                  << nrrcu << " broken rcus");
  }
  outfile.close();
}

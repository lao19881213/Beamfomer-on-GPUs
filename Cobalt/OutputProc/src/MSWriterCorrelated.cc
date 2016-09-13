//# MSWriterCorrelated.cc: a writer for correlated visibilities
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
//# $Id: MSWriterCorrelated.cc 25659 2013-07-13 13:10:10Z mol $

#include <lofar_config.h>

#include "MSWriterCorrelated.h"

#include <sys/types.h>
#include <fcntl.h>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <Common/SystemUtil.h>
#include <MSLofar/FailedTileInfo.h>
#include <CoInterface/CorrelatedData.h>

#include <tables/Tables/Table.h>
#include <casa/Quanta/MVTime.h>

#include "MeasurementSetFormat.h"

using boost::format;
using namespace casa;


namespace LOFAR
{
  namespace Cobalt
  {

    MSWriterCorrelated::MSWriterCorrelated (const std::string &logPrefix, const std::string &msName, const Parset &parset, unsigned subbandIndex, bool isBigEndian)
      :
      MSWriterFile(
        (makeMeasurementSet(logPrefix, msName, parset, subbandIndex, isBigEndian),
         str(format("%s/table.f0data") % msName))),
      itsLogPrefix(logPrefix),
      itsMSname(msName),
      itsParset(parset)
    {
      if (LofarStManVersion > 1) {
        string seqfilename = str(format("%s/table.f0seqnr") % msName);

        try {
          itsSequenceNumbersFile = new FileStream(seqfilename, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        } catch (...) {
          LOG_WARN_STR(itsLogPrefix << "Could not open sequence numbers file " << seqfilename);
        }
      }

#if 0
      // derive baseline names
      std::vector<std::string> stationNames = parset.mergedStationNames();
      std::vector<std::string> baselineNames(parset.nrBaselines());
      unsigned nrStations = stationNames.size();

      // order of baselines as station indices:
      // 0-0, 1-0, 1-1, 2-0, 2-1, 2-2 ... (see RTCP/CNProc/Correlator.cc)

      unsigned bl = 0;

      for(unsigned s1 = 0; s1 < nrStations; s1++)
        for(unsigned s2 = 0; s2 <= s1; s2++)
          //bl = s1 * (s1 + 1) / 2 + stat2 ;
          baselineNames[bl++] = str(format("%s_%s") % stationNames[s1] % stationNames[s2]);
#endif

      itsConfiguration.add("fileFormat",           "AIPS++/CASA");
      itsConfiguration.add("filename",             LOFAR::basename(msName));
      itsConfiguration.add("size",                 "0");
      itsConfiguration.add("location",             parset.getHostName(CORRELATED_DATA, subbandIndex) + ":" + parset.getDirectoryName(CORRELATED_DATA, subbandIndex));

      itsConfiguration.add("percentageWritten",    "0");
      itsConfiguration.add("startTime",            parset.getString("Observation.startTime"));
      itsConfiguration.add("duration",             "0");
      itsConfiguration.add("integrationInterval",  str(format("%f") % parset.IONintegrationTime()));
      itsConfiguration.add("centralFrequency",     str(format("%f") % parset.settings.subbands[subbandIndex].centralFrequency));
      itsConfiguration.add("channelWidth",         str(format("%f") % parset.channelWidth()));
      itsConfiguration.add("channelsPerSubband",   str(format("%u") % parset.nrChannelsPerSubband()));
      itsConfiguration.add("stationSubband",       str(format("%u") % parset.settings.subbands[subbandIndex].stationIdx));
      itsConfiguration.add("subband",              str(format("%u") % subbandIndex));
      itsConfiguration.add("SAP",                  str(format("%u") % parset.settings.subbands[subbandIndex].SAP));
    }


    MSWriterCorrelated::~MSWriterCorrelated()
    {
    }


    void MSWriterCorrelated::makeMeasurementSet(const std::string &logPrefix, const std::string &msName, const Parset &parset, unsigned subbandIndex, bool isBigEndian)
    {
#if defined HAVE_AIPSPP
      MeasurementSetFormat myFormat(parset, 512);

      myFormat.addSubband(msName, subbandIndex, isBigEndian);

      LOG_INFO_STR(logPrefix << "MeasurementSet created");
#endif // defined HAVE_AIPSPP
    }


    void MSWriterCorrelated::write(StreamableData *data)
    {
      CorrelatedData *cdata = dynamic_cast<CorrelatedData*>(data);

      ASSERT( data );
      ASSERT( cdata );

      // Write data
      MSWriterFile::write(data);

      // Write sequence number
      if (itsSequenceNumbersFile != 0) {
        // quick fix: always write to maintain integrity
        unsigned seqnr = data->sequenceNumber(true);

        itsSequenceNumbersFile->write(&seqnr, sizeof seqnr);
      }

      itsNrBlocksWritten++;

      itsConfiguration.replace("size",     str(format("%u") % getDataSize()));
      itsConfiguration.replace("duration", str(format("%f") % ((data->sequenceNumber() + 1) * itsParset.IONintegrationTime())));
      itsConfiguration.replace("percentageWritten", str(format("%u") % percentageWritten()));
    }


    static MVEpoch datetime2epoch(const string &datetime)
    {
      Quantity q;

      if (!MVTime::read(q, datetime))
        return MVEpoch(0);

      return MVEpoch(q);
    }


    void MSWriterCorrelated::augment(const FinalMetaData &finalMetaData)
    {
      ScopedLock sl(MeasurementSetFormat::sharedMutex);

      map<string, FailedTileInfo::VectorFailed> brokenBefore, brokenDuring;

      // fill set of broken hardware at beginning of observation
      for (size_t i = 0; i < finalMetaData.brokenRCUsAtBegin.size(); i++) {
        const struct FinalMetaData::BrokenRCU &rcu = finalMetaData.brokenRCUsAtBegin[i];

        brokenBefore[rcu.station].push_back(FailedTileInfo(rcu.station, rcu.time, datetime2epoch(rcu.time), rcu.type, rcu.seqnr));
      }

      // fill set of hardware that broke during the observation
      for (size_t i = 0; i < finalMetaData.brokenRCUsDuring.size(); i++) {
        const struct FinalMetaData::BrokenRCU &rcu = finalMetaData.brokenRCUsDuring[i];

        brokenDuring[rcu.station].push_back(FailedTileInfo(rcu.station, rcu.time, datetime2epoch(rcu.time), rcu.type, rcu.seqnr));
      }

      LOG_INFO_STR(itsLogPrefix << "Reopening MeasurementSet");

      Table ms(itsMSname, Table::Update);

      vector<FailedTileInfo::VectorFailed> before(FailedTileInfo::antennaConvert(ms, brokenBefore));
      vector<FailedTileInfo::VectorFailed> during(FailedTileInfo::antennaConvert(ms, brokenDuring));

      LOG_INFO_STR(itsLogPrefix << "Writing broken hardware information to MeasurementSet");

      try {
        FailedTileInfo::writeFailed(ms, before, during);
      } catch (Exception &ex) {
        LOG_ERROR_STR("Failed to write broken hardware information: " << ex);
      }
    }


  } // namespace Cobalt
} // namespace LOFAR


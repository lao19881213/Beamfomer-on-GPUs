//# MSWriterCorrelated: a writer for correlated visibilities
//#
//#  Copyright (C) 2001
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
#include <Interface/CorrelatedData.h>
#include <MSLofar/FailedTileInfo.h>
#include <Common/SystemUtil.h>
#include <Storage/MSWriterCorrelated.h>
#include <Storage/MeasurementSetFormat.h>
#include <tables/Tables/Table.h>
#include <casa/Quanta/MVTime.h>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/types.h>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
using boost::format;
using namespace casa;


namespace LOFAR {
namespace RTCP {

MSWriterCorrelated::MSWriterCorrelated (const std::string &logPrefix, const std::string &msName, const Parset &parset, unsigned subbandIndex, bool isBigEndian)
:
  MSWriterFile(
      (makeMeasurementSet(logPrefix, msName, parset, subbandIndex, isBigEndian), 
      str(format("%s/table.f0data") % msName))),
  itsLogPrefix(logPrefix),
  itsMSname(msName),
  itsParset(parset)
{
  if (itsParset.getLofarStManVersion() > 1) {
    string seqfilename = str(format("%s/table.f0seqnr") % msName);
    
    try {
      itsSequenceNumbersFile = new FileStream(seqfilename, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR |  S_IWUSR | S_IRGRP | S_IROTH);
    } catch (...) {
      LOG_WARN_STR(itsLogPrefix << "Could not open sequence numbers file " << seqfilename);
    }
  }

  // derive baseline names
  std::vector<std::string> stationNames = parset.mergedStationNames();
  std::vector<std::string> baselineNames(parset.nrBaselines());
  unsigned nrStations = stationNames.size();

  // order of baselines as station indices:
  // 0-0, 1-0, 1-1, 2-0, 2-1, 2-2 ... (see RTCP/CNProc/Correlator.cc)

  unsigned bl = 0;

  for(unsigned s1 = 0; s1 < nrStations; s1++)
    for(unsigned s2 = 0; s2 <= s1; s2++)
      baselineNames[bl++] = str(format("%s_%s") % stationNames[s1] % stationNames[s2]);

  const vector<unsigned> subbands  = itsParset.subbandList();
  const vector<unsigned> SAPs      = itsParset.subbandToSAPmapping();
  const vector<double> frequencies = itsParset.subbandToFrequencyMapping();

  itsConfiguration.add("fileFormat",           "AIPS++/CASA");
  itsConfiguration.add("filename",             LOFAR::basename(msName));
  itsConfiguration.add("size",                 "0");
  //itsConfiguration.add("location",             parset.getHostName(CORRELATED_DATA, subbandIndex) + ":" + LOFAR::dirname(msName));
  itsConfiguration.add("location",             parset.getHostName(CORRELATED_DATA, subbandIndex) + ":" + parset.getDirectoryName(CORRELATED_DATA, subbandIndex));

  itsConfiguration.add("percentageWritten",    "0");
  itsConfiguration.add("startTime",            parset.getString("Observation.startTime"));
  itsConfiguration.add("duration",             "0");
  itsConfiguration.add("integrationInterval",  str(format("%f") % parset.IONintegrationTime()));
  itsConfiguration.add("centralFrequency",     str(format("%f") % frequencies[subbandIndex]));
  itsConfiguration.add("channelWidth",         str(format("%f") % parset.channelWidth()));
  itsConfiguration.add("channelsPerSubband",   str(format("%u") % parset.nrChannelsPerSubband()));
  itsConfiguration.add("stationSubband",       str(format("%u") % subbands[subbandIndex]));
  itsConfiguration.add("subband",              str(format("%u") % subbandIndex));
  itsConfiguration.add("SAP",                  str(format("%u") % SAPs[subbandIndex]));
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

  FailedTileInfo::writeFailed(ms, before, during);
}


} // namespace RTCP
} // namespace LOFAR


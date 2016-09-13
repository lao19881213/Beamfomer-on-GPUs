//#  MeasurementSetFormat.cc: Creates required infrastructure for 
//#  a LofarStMan MeasurementSet.
//#
//#  Copyright (C) 2009
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  $Id: $


#include <lofar_config.h>

#include <Storage/MeasurementSetFormat.h>
#include <Storage/Package__Version.h>

#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <linux/limits.h>

#include <tables/Tables/TableDesc.h>
#include <tables/Tables/SetupNewTab.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/TableLock.h>
#include <tables/Tables/TableRecord.h>
#include <tables/Tables/ScaColDesc.h>
#include <tables/Tables/ArrColDesc.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/ArrayColumn.h>
//#include <tables/Tables/StandardStMan.h>
#include <casa/Arrays/Array.h>
#include <casa/Arrays/ArrayMath.h>
#include <casa/Arrays/ArrayIO.h>
#include <casa/Arrays/ArrayLogical.h>
#include <casa/Containers/BlockIO.h>
#include <casa/OS/RegularFile.h>
#include <casa/Utilities/Assert.h>
#include <casa/IO/RegularFileIO.h>
#include <casa/IO/RawIO.h>
#include <casa/IO/CanonicalIO.h>
#include <casa/OS/HostInfo.h>
#include <casa/Exceptions/Error.h>
#include <casa/iostream.h>
#include <casa/sstream.h>
#include <casa/BasicSL/Constants.h>

#include <ms/MeasurementSets.h>

#include <MSLofar/MSLofar.h>
#include <MSLofar/MSLofarField.h>
#include <MSLofar/MSLofarAntenna.h>
#include <MSLofar/MSLofarObservation.h>
#include <MSLofar/MSLofarAntennaColumns.h>
#include <MSLofar/MSLofarFieldColumns.h>
#include <MSLofar/MSLofarObsColumns.h>
#include <MSLofar/BeamTables.h>
#include <LofarStMan/LofarStMan.h>
#include <Interface/Exceptions.h>


using namespace casa;

namespace LOFAR { 
namespace RTCP {


Mutex MeasurementSetFormat::sharedMutex;


// unix time to mjd time (in seconds instead of days)
static double toMJDs( double time )
{
  // 40587 modify Julian day number = 00:00:00 January 1, 1970, GMT
  return 40587.0 * 24 * 60 * 60 + time;
}


MeasurementSetFormat::MeasurementSetFormat(const Parset &ps, unsigned alignment)
:
  itsPS(ps),
  stationNames(itsPS.mergedStationNames()),
  antPos(itsPS.positions()),
  itsNrAnt(stationNames.size()),
  itsMS(0), 
  itsAlignment(alignment)
{
  if (itsPS.nrTabStations() > 0) { 
    ASSERTSTR(antPos.size() == 3 * itsPS.nrTabStations(),
	      antPos.size() << " == " << 3 * itsPS.nrTabStations());
  } else {
    ASSERTSTR(antPos.size() == 3 * itsPS.nrStations(),
	      antPos.size() << " == " << 3 * itsPS.nrStations());
  }

  itsStartTime = toMJDs(itsPS.startTime());

  itsTimeStep = itsPS.IONintegrationTime();
  itsNrTimes = itsPS.nrCorrelatedBlocks();
}

  
MeasurementSetFormat::~MeasurementSetFormat()  
{
}


void MeasurementSetFormat::addSubband(const string MSname, unsigned subband, bool isBigEndian)
{
  ScopedLock scopedLock(sharedMutex);

  /// First create a valid MeasurementSet with all required
  /// tables. Note that the MS object is destroyed immediately.
  createMSTables(MSname, subband);
  /// Next make a metafile which describes the raw datafile we're
  /// going to write
  createMSMetaFile(MSname, subband, isBigEndian);
}


void MeasurementSetFormat::createMSTables(const string &MSname, unsigned subband)
{
  try {
    TableDesc td = MS::requiredTableDesc();
    MS::addColumnToDesc(td, MS::DATA, 2);
    MS::addColumnToDesc(td, MS::WEIGHT_SPECTRUM, 2);
    // Set the reference frame of UVW to J2000.
    // Note it must be done here, because the UVW column in the MS is readonly
    // (because LofarStMan is used).
    {
      ColumnDesc &col(td.rwColumnDesc("UVW"));
      TableRecord rec = col.keywordSet().asRecord("MEASINFO");
      rec.define("Ref", "J2000");
      col.rwKeywordSet().defineRecord("MEASINFO", rec);
    }

    SetupNewTable newtab(MSname, td, Table::New);
    LofarStMan lofarstman;
    newtab.bindAll(lofarstman);

    itsMS = new MSLofar(newtab);
    itsMS->createDefaultSubtables(Table::New);

    Block<MPosition> antMPos(itsNrAnt);

    try {
      for (unsigned i = 0; i < itsNrAnt; i ++) {
	antMPos[i] = MPosition(MVPosition(antPos[3 * i], 
					  antPos[3 * i + 1], 
					  antPos[3 * i + 2]),
			       MPosition::ITRF);
      }
    } catch (AipsError &ex) {
      LOG_FATAL_STR("AipsError: " << ex.what());
    }

    // Get subarray id (formerly known as beam).
    const vector<unsigned> subbandToSAPmapping = itsPS.subbandToSAPmapping();
    int subarray = subbandToSAPmapping[subband]; 

    fillAntenna(antMPos);
    fillFeed();
    fillField(subarray);
    fillPola();
    fillDataDesc();
    fillSpecWindow(subband);
    fillObs(subarray);
    fillHistory();

    try {
      // Fill the tables containing the beam info.
      BeamTables::fill(*itsMS,
		       itsPS.antennaSet(),
                       itsPS.AntennaSetsConf(),
                       itsPS.AntennaFieldsDir(),
                       itsPS.HBADeltasDir());
    } catch (LOFAR::AssertError &ex) {
      LOG_WARN_STR("Ignoring exception from BeamTables::fill(): " << ex.what());
    }
  } catch (AipsError &ex) {
    THROW(StorageException, "AIPS/CASA error: " << ex.getMesg());
  }

  // Flush the MS to make sure all tables are written
  itsMS->flush();
  // Delete the MS object, since we don't need it anymore
}


void MeasurementSetFormat::fillAntenna(const Block<MPosition>& antMPos)
{
  // Determine constants for the ANTENNA subtable.
  casa::Vector<Double> antOffset(3);
  antOffset = 0;
  casa::Vector<Double> phaseRef(3);

  // Fill the ANTENNA subtable.
  MSLofarAntenna msant = itsMS->antenna();
  MSLofarAntennaColumns msantCol(msant);
  msant.addRow (itsNrAnt);
      
  for (unsigned i = 0; i < itsNrAnt; i ++) {
    msantCol.name().put(i, stationNames[i]);
    msantCol.stationId().put(i, 0);
    msantCol.station().put(i, "LOFAR");
    msantCol.type().put(i, "GROUND-BASED");
    msantCol.mount().put(i, "X-Y");
    msantCol.positionMeas().put(i, antMPos[i]);
    msantCol.offset().put(i, antOffset);
    msantCol.dishDiameter().put(i, 0);
    vector<double> psPhaseRef =
      itsPS.getDoubleVector("PIC.Core."+stationNames[i]+".phaseCenter");
    ASSERTSTR(psPhaseRef.size() == 3,
	      "phaseCenter in parset of station " << stationNames[i]);
    std::copy(psPhaseRef.begin(), psPhaseRef.end(), phaseRef.begin());
    msantCol.phaseReference().put(i, phaseRef);
    msantCol.flagRow().put(i, False);
  }

  msant.flush();
}


void MeasurementSetFormat::fillFeed()
{
  // Determine constants for the FEED subtable.
  unsigned nRec = 2;
  casa::Matrix<Double> feedOffset(2,nRec);
  feedOffset = 0;
  casa::Matrix<Complex> feedResponse(nRec,nRec);
  feedResponse = Complex(0.0,0.0);

  for (unsigned rec = 0; rec < nRec; rec ++)
    feedResponse(rec,rec) = Complex(1.0, 0.0);

  casa::Vector<String> feedType(nRec);
  feedType(0) = "X";
  feedType(1) = "Y";
  casa::Vector<Double> feedPos(3);
  feedPos = 0.0;
  casa::Vector<Double> feedAngle(nRec);
  feedAngle = -C::pi_4;                      // 0 for parallel dipoles

  // Fill the FEED subtable.
  MSFeed msfeed = itsMS->feed();
  MSFeedColumns msfeedCol(msfeed);
  msfeed.addRow(itsNrAnt);

  for (unsigned i = 0; i < itsNrAnt; i ++) {
    msfeedCol.antennaId().put(i, i);
    msfeedCol.feedId().put(i, 0);
    msfeedCol.spectralWindowId().put(i, -1);
    msfeedCol.time().put(i, itsStartTime + itsNrTimes * itsTimeStep / 2.);
    msfeedCol.interval().put(i, itsNrTimes * itsTimeStep);
    msfeedCol.beamId().put(i, -1);
    msfeedCol.beamOffset().put(i, feedOffset);
    msfeedCol.polarizationType().put(i, feedType);
    msfeedCol.polResponse().put(i, feedResponse);
    msfeedCol.position().put(i, feedPos);
    msfeedCol.receptorAngle().put(i, feedAngle);
    msfeedCol.numReceptors().put(i, 2);
  }

  msfeed.flush();
}


void MeasurementSetFormat::fillField(unsigned subarray)
{

  // Beam direction
  MVDirection radec(Quantity(itsPS.getBeamDirection(subarray)[0], "rad"), 
		    Quantity(itsPS.getBeamDirection(subarray)[1], "rad"));
  MDirection::Types beamDirectionType;
  MDirection::getType(beamDirectionType, itsPS.getBeamDirectionType(subarray));
  MDirection indir(radec, beamDirectionType);
  casa::Vector<MDirection> outdir(1);
  outdir(0) = indir;

  // AnaBeam direction type
  MDirection::Types anaBeamDirectionType;
  if (itsPS.haveAnaBeam())
    MDirection::getType(anaBeamDirectionType, itsPS.getAnaBeamDirectionType());

  // Put the direction into the FIELD subtable.
  MSLofarField msfield = itsMS->field();
  MSLofarFieldColumns msfieldCol(msfield);

  uInt rownr = msfield.nrow();
  ASSERT(rownr == 0); // can only set directionType on first row, so only one field per MeasurementSet for now

  if (itsPS.haveAnaBeam())
    msfieldCol.setDirectionRef(beamDirectionType, anaBeamDirectionType);
  else
    msfieldCol.setDirectionRef(beamDirectionType);

  msfield.addRow();
  msfieldCol.name().put(rownr, "BEAM_" + String::toString(subarray));
  msfieldCol.code().put(rownr, "");
  msfieldCol.time().put(rownr, itsStartTime);
  msfieldCol.numPoly().put(rownr, 0);

  msfieldCol.delayDirMeasCol().put(rownr, outdir);
  msfieldCol.phaseDirMeasCol().put(rownr, outdir);
  msfieldCol.referenceDirMeasCol().put(rownr, outdir);

  msfieldCol.sourceId().put(rownr, -1);
  msfieldCol.flagRow().put(rownr, False);

  if (itsPS.haveAnaBeam()) {
    // Analog beam direction
    MVDirection radec_AnaBeamDirection(Quantity(itsPS.getAnaBeamDirection()[0], "rad"),
  				       Quantity(itsPS.getAnaBeamDirection()[1], "rad"));
    MDirection anaBeamDirection(radec_AnaBeamDirection, anaBeamDirectionType);
    msfieldCol.tileBeamDirMeasCol().put(rownr, anaBeamDirection);
  } else {
    msfieldCol.tileBeamDirMeasCol().put(rownr, outdir(0));
  }
}


void MeasurementSetFormat::fillPola()
{
  const unsigned npolarizations = itsPS.nrCrossPolarisations();

  MSPolarization mspol = itsMS->polarization();
  MSPolarizationColumns mspolCol(mspol);
  uInt rownr = mspol.nrow();
  casa::Vector<Int> corrType(npolarizations);
  corrType(0) = Stokes::XX;

  if (npolarizations == 2) {
    corrType(1) = Stokes::YY;
  } else if (npolarizations == 4) {
    corrType(1) = Stokes::XY;
    corrType(2) = Stokes::YX;
    corrType(3) = Stokes::YY;
  }

  casa::Matrix<Int> corrProduct(2, npolarizations);

  for (unsigned i = 0; i < npolarizations; i++) {
    corrProduct(0,i) = Stokes::receptor1(Stokes::type(corrType(i)));
    corrProduct(1,i) = Stokes::receptor2(Stokes::type(corrType(i)));
  }

  // Fill the columns.
  mspol.addRow();
  mspolCol.numCorr().put(rownr, npolarizations);
  mspolCol.corrType().put(rownr, corrType);
  mspolCol.corrProduct().put(rownr, corrProduct);
  mspolCol.flagRow().put(rownr, False);
  mspol.flush();
}


void MeasurementSetFormat::fillDataDesc()
{
  MSDataDescription msdd = itsMS->dataDescription();
  MSDataDescColumns msddCol(msdd);
  
  msdd.addRow();

  msddCol.spectralWindowId().put(0, 0);
  msddCol.polarizationId().put(0, 0);
  msddCol.flagRow().put(0, False);

  msdd.flush();
}


void MeasurementSetFormat::fillObs(unsigned subarray)
{
  // Get start and end time.
  casa::Vector<Double> timeRange(2);
  timeRange[0] = itsStartTime;
  timeRange[1] = itsStartTime + itsNrTimes*itsTimeStep;

  // Get minimum and maximum frequency.
  vector<double> freqs = itsPS.subbandToFrequencyMapping();
  ASSERT( freqs.size() > 0 );

  double minFreq = *std::min_element( freqs.begin(), freqs.end() );
  double maxFreq = *std::max_element( freqs.begin(), freqs.end() );

  size_t nchan = itsPS.nrChannelsPerSubband();

  if( nchan > 1 ) {
    // 2nd PPF shifts frequencies downwards by half a channel
    double width = itsPS.channelWidth();

    minFreq -= 0.5 * nchan * width;
    maxFreq -= 0.5 * nchan * width;
  }

  casa::Vector<String> corrSchedule(1);
  corrSchedule = "corrSchedule";

  vector<string> targets(itsPS.getStringVector
	 ("Observation.Beam[" + String::toString(subarray) + "].target"));
  casa::Vector<String> ctargets(targets.size());

  for (uint i = 0; i < targets.size(); ++ i)
    ctargets[i] = targets[i];

  vector<string> cois(itsPS.getStringVector("Observation.Campaign.CO_I"));
  casa::Vector<String> ccois(cois.size());

  for (uint i = 0; i < cois.size(); ++ i)
    ccois[i] = cois[i];
			  
  double releaseDate = timeRange(1) + 365.25 * 24 * 60 * 60;

  MSLofarObservation msobs = itsMS->observation();
  MSLofarObservationColumns msobsCol(msobs);

  msobs.addRow();

  msobsCol.telescopeName().put(0, "LOFAR");
  msobsCol.timeRange().put(0, timeRange);
  msobsCol.observer().put(0, "unknown");
  msobsCol.scheduleType().put(0, "LOFAR");
  msobsCol.schedule().put(0, corrSchedule);
  msobsCol.project().put(0, itsPS.getString("Observation.Campaign.name"));
  msobsCol.releaseDate().put(0, releaseDate);
  msobsCol.flagRow().put(0, False);
  msobsCol.projectTitle().put(0, itsPS.getString("Observation.Campaign.title"));
  msobsCol.projectPI().put(0,  itsPS.getString("Observation.Campaign.PI"));
  msobsCol.projectCoI().put(0, ccois);
  msobsCol.projectContact().put(0, itsPS.getString("Observation.Campaign.contact"));
  msobsCol.observationId().put(0, String::toString(itsPS.observationID()));
  msobsCol.observationStart().put(0, timeRange[0]);
  msobsCol.observationEnd().put(0, timeRange[1]);
  msobsCol.observationFrequencyMaxQuant().put(0, Quantity(maxFreq, "Hz"));
  msobsCol.observationFrequencyMinQuant().put(0, Quantity(minFreq, "Hz"));
  msobsCol.observationFrequencyCenterQuant().put(0, Quantity(0.5*(minFreq+maxFreq), "Hz"));
  msobsCol.subArrayPointing().put(0, subarray);
  msobsCol.nofBitsPerSample().put(0, itsPS.nrBitsPerSample());
  msobsCol.antennaSet().put(0, itsPS.antennaSet());
  msobsCol.filterSelection().put(0, itsPS.bandFilter());
  msobsCol.clockFrequencyQuant().put(0, Quantity(itsPS.clockSpeed(), "Hz"));
  msobsCol.target().put(0, ctargets);
  msobsCol.systemVersion().put(0, Version::getInfo<StorageVersion>("Storage",
								   "brief"));
  msobsCol.pipelineName().put(0, String());
  msobsCol.pipelineVersion().put(0, String());
  msobsCol.filename().put(0, Path(itsMS->tableName()).baseName());
  msobsCol.filetype().put(0, "uv");
  msobsCol.filedate().put(0, timeRange[0]);

  msobs.flush();
}

void MeasurementSetFormat::fillSpecWindow(unsigned subband) {
  const double refFreq   = itsPS.subbandToFrequencyMapping()[subband];
  const size_t nchan     = itsPS.nrChannelsPerSubband();
  const double chanWidth = itsPS.channelWidth();
  const double totalBW   = nchan * chanWidth;
  const double channel0freq = itsPS.channel0Frequency(subband);

  casa::Vector<double> chanWidths(nchan, chanWidth);
  casa::Vector<double> chanFreqs(nchan);
  indgen (chanFreqs, channel0freq, chanWidth);

  MSSpectralWindow msspw = itsMS->spectralWindow();
  MSSpWindowColumns msspwCol(msspw);
    
  msspw.addRow();

  msspwCol.numChan().put(0, nchan);
  msspwCol.name().put(0, "SB-" + String::toString(subband));
  msspwCol.refFrequency().put(0, refFreq);
  msspwCol.chanFreq().put(0, chanFreqs);

  msspwCol.chanWidth().put(0, chanWidths);
  msspwCol.measFreqRef().put(0, MFrequency::TOPO);
  msspwCol.effectiveBW().put(0, chanWidths);
  msspwCol.resolution().put(0, chanWidths);
  msspwCol.totalBandwidth().put(0, totalBW);
  msspwCol.netSideband().put(0, 0);
  msspwCol.ifConvChain().put(0, 0);
  msspwCol.freqGroup().put(0, 0);
  msspwCol.freqGroupName().put(0, "");
  msspwCol.flagRow().put(0, False);

  msspw.flush();
}


void MeasurementSetFormat::fillHistory()
{
  Table histtab(itsMS->keywordSet().asTable("HISTORY")); 
  histtab.reopenRW(); 
  ScalarColumn<double> time        (histtab, "TIME"); 
  ScalarColumn<int>    obsId       (histtab, "OBSERVATION_ID"); 
  ScalarColumn<String> message     (histtab, "MESSAGE"); 
  ScalarColumn<String> application (histtab, "APPLICATION"); 
  ScalarColumn<String> priority    (histtab, "PRIORITY"); 
  ScalarColumn<String> origin      (histtab, "ORIGIN"); 
  ArrayColumn<String>  parms       (histtab, "APP_PARAMS"); 
  ArrayColumn<String>  cli         (histtab, "CLI_COMMAND"); 

  // Put all parset entries in a Vector<String>. 
  casa::Vector<String> appvec; 
  casa::Vector<String> clivec; 
  appvec.resize (itsPS.size()); 
  casa::Array<String>::contiter viter = appvec.cbegin(); 
  for (ParameterSet::const_iterator iter = itsPS.begin(); iter != itsPS.end(); ++iter, ++viter) { 
    *viter = iter->first + '=' + iter->second.get(); 
  } 
  uint rownr = histtab.nrow(); 
  histtab.addRow(); 
  time.put        (rownr, Time().modifiedJulianDay()*24.*3600.); 
  obsId.put       (rownr, 0); 
  message.put     (rownr, "parameters");
  application.put (rownr, "OLAP"); 
  priority.put    (rownr, "NORMAL"); 
  origin.put      (rownr, Version::getInfo<StorageVersion>("Storage", "full")); 
  parms.put       (rownr, appvec); 
  cli.put         (rownr, clivec); 
}


void MeasurementSetFormat::createMSMetaFile(const string &MSname, unsigned subband, bool isBigEndian)
{ 
  (void) subband;

  Block<Int> ant1(itsPS.nrBaselines());
  Block<Int> ant2(itsPS.nrBaselines());
  uInt inx = 0;
  uInt nStations = itsPS.nrTabStations() > 0 ? itsPS.nrTabStations() : itsPS.nrStations();

  for (uInt i = 0; i < nStations; ++ i) {
    for (uInt j = 0; j <= i; ++ j) {

      if (itsPS.getLofarStManVersion() == 1) {
	ant1[inx] = j;
	ant2[inx] = i;
	++ inx;
      } else {
	// switch order of stations to fix write of complex conjugate data in V1
	ant1[inx] = i;
	ant2[inx] = j;
	++ inx;
      }
    }
  }

  string filename = MSname + "/table.f0meta";
  
  AipsIO aio(filename, ByteIO::New);
  aio.putstart("LofarStMan", itsPS.getLofarStManVersion()); 
  aio << ant1 << ant2
      << itsStartTime
      << itsPS.IONintegrationTime()
      << itsPS.nrChannelsPerSubband()
      << itsPS.nrCrossPolarisations()
      << static_cast<double>(itsPS.CNintegrationSteps() * itsPS.IONintegrationSteps())
      << itsAlignment
      << isBigEndian;
  if (itsPS.getLofarStManVersion() > 1) {
    uInt itsNrBytesPerNrValidSamples = 
      itsPS.integrationSteps() < 256 ? 1 : itsPS.integrationSteps() < 65536 ? 2 : 4;
    aio << itsNrBytesPerNrValidSamples;
  }
  aio.close();
}
 

} // namespace RTCP
} // namepsace LOFAR

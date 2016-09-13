//# TBB_Writer.cc: Write TBB data into an HDF5 file
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: TBB_Writer.cc 25574 2013-07-04 16:02:41Z amesfoort $

#include <lofar_config.h>

#ifdef HAVE_DAL

#define _FILE_OFFSET_BITS 64
#include <cstddef>
#include <cstring>
#include <csignal>
#include <ctime>
#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <endian.h>
#if __BYTE_ORDER != __BIG_ENDIAN && __BYTE_ORDER != __LITTLE_ENDIAN
#error Byte order is neither big endian nor little endian: not supported
#endif

#include <iostream>
#include <sstream>
#include <algorithm>

#include <OutputProc/TBB_Writer.h>
#include <Common/LofarConstants.h>
#include <Common/LofarLogger.h>
#ifdef basename // some glibc have this as a macro
#undef basename
#endif
#include <Common/SystemUtil.h>
#include <Common/SystemCallException.h>
#include <Common/StringUtil.h>
#include <Common/StreamUtil.h>
#include <ApplCommon/AntField.h>
#include <Stream/SocketStream.h>
#include <CoInterface/Exceptions.h>
#include <CoInterface/Stream.h>

#include <dal/lofar/StationNames.h>

#define TBB_TRANSIENT_MODE                      1
#define TBB_SPECTRAL_MODE                       2

#define RSP_NR_SUBBANDS                         512

namespace LOFAR
{
  namespace Cobalt
  {

    using namespace std;

    EXCEPTION_CLASS(TBB_MalformedFrameException, StorageException);

    // The output_format is without seconds. The output_size is including the terminating NUL char.
    static string formatFilenameTimestamp(const struct timeval& tv, const char* output_format,
                                          const char* output_format_secs, size_t output_size)
    {
      struct tm tm;
      ::gmtime_r(&tv.tv_sec, &tm);
      double secs = tm.tm_sec + tv.tv_usec / 1000000.0;

      vector<char> date(output_size);

      size_t nwritten = ::strftime(&date[0], output_size, output_format, &tm);
      if (nwritten == 0) {
        date[0] = '\0';
      }
      (void)::snprintf(&date[0] + nwritten, output_size - nwritten, output_format_secs, secs);

      return string(&date[0]);
    }

    // FileStream doesn't do pwrite(2). Nobody else needs it, so define it here, but in the same way.
    static size_t tryPWrite(int fd, const void *ptr, size_t size, off_t offset)
    {
      ssize_t bytes = ::pwrite(fd, ptr, size, offset);
      if (bytes < 0)
        THROW_SYSCALL("pwrite");
      return bytes;
    }

    static void pwrite(int fd, const void *ptr, size_t size, off_t offset)
    {
      while (size > 0) {
        size_t bytes = tryPWrite(fd, ptr, size, offset);
        size -= bytes;
        offset += bytes;
        ptr = static_cast<const char *>(ptr) + bytes;
      }
    }

    static ostream& operator<<(ostream& out, const TBB_Header& h)
    {
      out << (unsigned)h.stationID << " " << (unsigned)h.rspID << " " << (unsigned)h.rcuID << " " << (unsigned)h.sampleFreq <<
      " " << h.seqNr << " " << h.time << " " << (h.nOfFreqBands == 0 ? h.sampleNr : h.bandSliceNr) << " " << h.nOfSamplesPerFrame <<
      " " << h.nOfFreqBands << " " << h.spare << " " << h.crc16; // casts uin8_t to unsigned to avoid printing as char
      return out;
    }

//////////////////////////////////////////////////////////////////////////////

    TBB_Dipole::TBB_Dipole()
      : itsRawOut(NULL) // needed, setting the others is superfluous
      , itsDataset(NULL)
      , itsFlagOffsets()
      , itsSampleFreq(0)
      , itsNrSubbands(0)
      , itsTime(0)
      , itsExpSampleNr(0)
      , itsDatasetLen(0)
    {
    }

    // Do not use. Only needed for vector<TBB_Dipole>(N).
    TBB_Dipole::TBB_Dipole(const TBB_Dipole& rhs)
      : itsRawOut(NULL) // idem. FileStream has no copy constr, but only copied before really set, so NULL is fine.
      , itsDataset(rhs.itsDataset)
      , itsFlagOffsets(rhs.itsFlagOffsets)
      , itsSampleFreq(rhs.itsSampleFreq)
      , itsNrSubbands(rhs.itsNrSubbands)
      , itsTime(rhs.itsTime)
      , itsExpSampleNr(rhs.itsExpSampleNr)
      , itsDatasetLen(rhs.itsDatasetLen)
    {
    }

    TBB_Dipole::~TBB_Dipole()
    {
      // Executed by the main thread after joined with all workers, so no need to lock or delay cancellation.
      if (isInitialized()) {
        try {
          if (itsNrSubbands == 0) { // transient mode
            itsDataset->resize1D(itsDatasetLen);
          } else { // spectral mode
            vector<ssize_t> newDims(2);
            newDims[0] = itsDatasetLen;
            newDims[1] = itsNrSubbands; // only the 1st dim can be extended
            itsDataset->resize(newDims);
          }
        } catch (exception& exc) { // dal::DALException, or std::bad_alloc from vector constr
          LOG_WARN_STR("TBB: failed to resize HDF5 dipole dataset to external data size: " << exc.what());
        }

        try {
          itsDataset->dataLength().value = static_cast<unsigned long long>(itsDatasetLen);
        } catch (dal::DALException& exc) {
          LOG_WARN_STR("TBB: failed to set dipole DATA_LENGTH attribute: " << exc.what());
        }
        try {
          itsDataset->flagOffsets().create(itsFlagOffsets.size()).set(itsFlagOffsets);
        } catch (dal::DALException& exc) {
          LOG_WARN_STR("TBB: failed to set dipole FLAG_OFFSETS attribute: " << exc.what());
        }

        delete itsDataset;
        delete itsRawOut;
      }
    }

    void TBB_Dipole::init(const TBB_Header& header, const Parset& parset,
                          const StationMetaData& stationMetaData,
                          const SubbandInfo& subbandInfo, const string& rawFilename,
                          dal::TBB_Station& station, Mutex& h5Mutex)
    {
      itsSampleFreq = static_cast<uint32_t>(header.sampleFreq) * 1000000;
      itsNrSubbands = header.nOfFreqBands;
      if (itsNrSubbands > subbandInfo.centralFreqs.size()) {
        throw StorageException("TBB: dropping frame with invalid nOfFreqBands");
      }

      itsRawOut = new FileStream(rawFilename, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);

      {
        ScopedLock h5OutLock(h5Mutex);
        try {
          initTBB_DipoleDataset(header, parset, stationMetaData, subbandInfo, rawFilename, station);
        } catch (exception& ) {
          /*
           * This nonsense is needed, because FileStream has no FileStream() and open() (and swap()),
           * and since we know the filename only at runtime (timestamp), we need itsRawOut to be a raw ptr.
           * We already have a raw ptr for the dataset and >1 raw ptr in 1 C++ class becomes buggy or messy.
           */
          delete itsRawOut;
          itsRawOut = NULL;
          throw;
        }
      }

      itsTime = header.time;
      if (itsNrSubbands == 0) { // transient mode
        itsExpSampleNr = header.sampleNr;
      } else { // spectral mode
        itsExpSliceNr = header.bandSliceNr >> TBB_SLICE_NR_SHIFT;
      }
      itsDatasetLen = 0; // already 0, for completeness
    }

    bool TBB_Dipole::isInitialized() const
    {
      return itsRawOut != NULL;
    }

    // Add a new flag range at the end or extend the last stored flag range. 'len' may not be 0.
    void TBB_Dipole::appendFlags(size_t offset, size_t len)
    {
      if (itsFlagOffsets.empty() || offset > itsFlagOffsets.back().end) {
        itsFlagOffsets.push_back(dal::Range(offset, offset + len));
      } else { // extend
        itsFlagOffsets.back().end += len;
      }
    }

    void TBB_Dipole::processTransientFrameData(const TBB_Frame& frame)
    {
      /*
       * Out-of-order or duplicate frames are very unlikely in the LOFAR TBB setup,
       * but let us know if it ever happens, then we will adapt this code and appendFlags().
       */
      if (frame.header.time < itsTime || (frame.header.time == itsTime && frame.header.sampleNr < itsExpSampleNr)) {
        LOG_WARN_STR("TBB: Unhandled out-of-order or duplicate frame: " <<
                     (unsigned)frame.header.stationID << " " << (unsigned)frame.header.rspID << " " << (unsigned)frame.header.rcuID <<
                     " " << frame.header.time << " " << itsTime << " " << frame.header.sampleNr << " " << itsExpSampleNr);
        return;
      }

      off_t offset = 0;
      if (frame.header.time == itsTime) {
        offset = itsDatasetLen + frame.header.sampleNr - itsExpSampleNr;
      } else { // crossed a seconds boundary, potentially more than once on excessive frame loss
        // A dump does not have to start at a sec bound, so up till the first bound, we may have had fewer than itsSampleFreq samples.
        if (itsDatasetLen < (int32_t)itsSampleFreq) {
          offset = itsDatasetLen;
          itsTime++;
        }
        offset += (off_t)(frame.header.time - itsTime) * itsSampleFreq;

        uint32_t newSecSampleNr0 = frame.header.sampleNr & (frame.header.nOfSamplesPerFrame - 1); // 0, or 512 by correctSampleNr()
        offset += frame.header.sampleNr - newSecSampleNr0;
      }

      /*
       * Flag lost frame(s) (assume no out-of-order, see below). Assumes all frames have the same nr of samples.
       * This cannot detect lost frames at the end of a dataset.
       */
      size_t nskipped = offset - itsDatasetLen;
      if (nskipped > 0) {
        appendFlags(itsDatasetLen, nskipped);
        itsRawOut->skip(nskipped * sizeof(frame.payload.data[0])); // skip space of lost frame(s)
      }

      /*
       * On a data checksum error, flag these samples.
       * Flag zeroed payloads too, as incredibly unlikely to be correct, but not rejected by crc32tbb.
       */
      if (!crc32tbb(&frame.payload, frame.header.nOfSamplesPerFrame)) {
        appendFlags(offset, frame.header.nOfSamplesPerFrame);
        uint32_t crc32;
        memcpy(&crc32, &frame.payload.data[frame.header.nOfSamplesPerFrame], sizeof crc32); // strict-aliasing safe
        LOG_WARN_STR("TBB: crc32: " << frame.header << " " << crc32);
      } else if (hasAllZeroDataSamples(frame.payload, frame.header.nOfSamplesPerFrame)) {
        appendFlags(offset, frame.header.nOfSamplesPerFrame);
      }

      // Since we are writing around HDF5, there is no need to lock. Resize the HDF5 dataset at the end (destr).
      itsRawOut->write(frame.payload.data, static_cast<size_t>(frame.header.nOfSamplesPerFrame) * sizeof(frame.payload.data[0]));

      itsTime = frame.header.time;
      itsExpSampleNr = frame.header.sampleNr + frame.header.nOfSamplesPerFrame;
      itsDatasetLen = offset + frame.header.nOfSamplesPerFrame;
    }

    void TBB_Dipole::processSpectralFrameData(const TBB_Frame& frame, const SubbandInfo& subbandInfo)
    {
      /*
       * Out-of-order or duplicate frames are very unlikely in the LOFAR TBB setup,
       * but let us know if it ever happens, then we will adapt this code and appendFlags().
       */
      uint32_t sliceNr = frame.header.bandSliceNr >> TBB_SLICE_NR_SHIFT; // cannot sanitize fully: too large values indicate lost data: flag
      if (frame.header.time < itsTime || (frame.header.time == itsTime && sliceNr < itsExpSliceNr)) {
        LOG_WARN_STR("TBB: Unhandled out-of-order or duplicate frame: " <<
                     (unsigned)frame.header.stationID << " " << (unsigned)frame.header.rspID << " " << (unsigned)frame.header.rcuID <<
                     " " << frame.header.time << " " << itsTime << " " << frame.header.bandSliceNr << " " << itsExpSliceNr);
        return;
      }

      off_t offset = 0;
      if (frame.header.time == itsTime) {
        offset = itsDatasetLen + sliceNr - itsExpSliceNr;
      } else { // crossed a seconds boundary, potentially more than once on excessive frame loss
        // A dump does not have to start at a sec bound, so up till the first bound, we may have had fewer than itsSampleFreq samples.
        if (itsDatasetLen < (int32_t)itsSampleFreq) {
          offset = itsDatasetLen;
          itsTime++;
        }
        offset += (off_t)(frame.header.time - itsTime) * itsSampleFreq + sliceNr;
      }

      /*
       * Flag lost frame(s) (assume no out-of-order, see below). Assumes all frames have the same nr of samples (fine).
       * This cannot detect lost frames at the end of a dataset.
       */
      size_t nskipped = offset - itsDatasetLen;
      if (nskipped > 0) {
        appendFlags(itsDatasetLen, nskipped); // no need to skip/lseek; we use pwrite() below
      }

      /*
       * On a data checksum error, flag these samples.
       * Flag zeroed payloads too, as incredibly unlikely to be correct, but not rejected by crc32tbb.
       *
       * TBB Design Doc states the crc32 is computed for transient data only, but it is also valid for spectral data.
       * Except that it looks invalid for the first spectral frame each second, so skip checking those. // TODO: enable 'sliceNr != 0 && ' below after verifying with recent real data
       */
      unsigned nSamplesPerSubband = frame.header.nOfSamplesPerFrame / itsNrSubbands; // any remainder is zeroed until the crc32
      if (/*sliceNr != 0 && */ !crc32tbb(&frame.payload, 2 * MAX_TBB_SPECTRAL_NSAMPLES)) {
        appendFlags(offset, nSamplesPerSubband);
        uint32_t crc32;
        memcpy(&crc32, &frame.payload.data[2 * MAX_TBB_SPECTRAL_NSAMPLES], sizeof crc32); // strict-aliasing safe
        LOG_WARN_STR("TBB: crc32: " << frame.header << " " << crc32);
      } else if (hasAllZeroDataSamples(frame.payload, 2 * frame.header.nOfSamplesPerFrame)) {
        appendFlags(offset, nSamplesPerSubband);
      }

      /*
       * In practice, each frame contains the same number of samples for all subbands, so the received band number is always 0.
       * Hence, disable support for cross-frame slices, such that in spectral mode we can also store flags in 1D.
       */
      /*unsigned bandNr = frame.header.bandSliceNr & TBB_BAND_NR_MASK;
         if (bandNr + itsNrSubbands >= RSP_NR_SUBBANDS) {
              LOG_WARN("TBB: Incorrect band number has been corrected to 0");
              bandNr = 0; // safe default
         }*/
      // Data arrives interleaved, so reorder, one sample at a time. Esp. inefficient if only 1 subband, but fast enough.
      for (unsigned i = 0; i < nSamplesPerSubband; ++i) {
        for (unsigned j = 0; j < itsNrSubbands; ++j) {
          off_t sampleOffset = (offset + subbandInfo.storageIndices[j /*(bandNr + j) % itsNrSubbands*/] * SPECTRAL_TRANSFORM_SIZE) * 2 * sizeof(frame.payload.data[0]);
          pwrite(itsRawOut->fd, &frame.payload.data[2 * (i * itsNrSubbands + j)], 2 * sizeof(frame.payload.data[0]), sampleOffset);
        }
        offset += 1;
      }

      itsTime = frame.header.time;
      itsExpSliceNr = sliceNr + nSamplesPerSubband;
      itsDatasetLen = offset;
    }

    void TBB_Dipole::initTBB_DipoleDataset(const TBB_Header& header, const Parset& parset,
                                           const StationMetaData& stationMetaData,
                                           const SubbandInfo& subbandInfo,
                                           const string& rawFilename, dal::TBB_Station& station)
    {
      // Override endianess. TBB data is always stored little endian and also received as such, so written as-is on any platform.
      if (subbandInfo.centralFreqs.empty()) { // transient mode
        dal::TBB_DipoleDataset* dpDataset = new dal::TBB_DipoleDataset(station.dipole(header.stationID, header.rspID, header.rcuID));
        itsDataset = static_cast<dal::TBB_Dataset<short>*>(dpDataset);

        itsDataset->create1D(0, -1, LOFAR::basename(rawFilename), itsDataset->LITTLE);

        dpDataset->sampleNumber().value = header.sampleNr;
      } else { // spectral mode
        dal::TBB_SubbandsDataset* sbDataset = new dal::TBB_SubbandsDataset(station.subbands(header.stationID, header.rspID, header.rcuID));
        itsDataset = reinterpret_cast<dal::TBB_Dataset<short>*>(sbDataset); // not so nice

        vector<ssize_t> dims(2), maxdims(2);
        dims[0] = 0;
        dims[1] = itsNrSubbands;
        maxdims[0] = -1; // only the 1st dim can be extendible
        maxdims[1] = itsNrSubbands;
        itsDataset->create(dims, maxdims, LOFAR::basename(rawFilename), itsDataset->LITTLE);

        sbDataset->sliceNumber().value = header.bandSliceNr >> TBB_SLICE_NR_SHIFT;
        sbDataset->spectralNofBands().value = itsNrSubbands;
        sbDataset->spectralBands().create(itsNrSubbands).set(subbandInfo.centralFreqs);
        sbDataset->spectralBandsUnit().value = "MHz";
      }

      itsDataset->groupType().value = "DipoleDataset";
      itsDataset->stationID().value = header.stationID;
      itsDataset->rspID().value = header.rspID;
      itsDataset->rcuID().value = header.rcuID;

      itsDataset->sampleFrequency().value = header.sampleFreq;
      itsDataset->sampleFrequencyUnit().value = "MHz";

      itsDataset->time().value = header.time; // in seconds

      itsDataset->samplesPerFrame().value = header.nOfSamplesPerFrame; // possibly sanitized
      //itsDataset->dataLength().value is set at the end (destr)
      //itsDataset->flagOffsets().value is set at the end (destr) // TODO: attrib -> 1D dataset
      itsDataset->nyquistZone().value = parset.nyquistZone();

      //#include "MAC/APL/PIC/RSP_Driver/src/CableSettings.h" or "RCUCables.h"
      // Cable delays (optional) from static meta data.
      //itsDataset->cableDelay().value = ???; // TODO
      //itsDataset->cableDelayUnit().value = "ns";

      /*
         > No DIPOLE_CALIBRATION_DELAY_VALUE
         > No DIPOLE_CALIBRATION_DELAY_UNIT
         These can be calculated from the values in the LOFAR calibration
         tables, but we can do that ourselves as long as the calibration table
         values for each dipole are written to the new keyword. Sander: please put them in; see the code ref below.
         DIPOLE_CALIBRATION_GAIN_CURVE.

         // Use StaticMetaData/CalTables

         calibration delay value en unit zijn nuttiger
         en is het beste om die er gelijk in te schrijven
         momenteel
         In /opt/cep/lus/daily/Mon/src/code/src/PyCRTools/modules/metadata.py
         heb ik code om de calibratie tabellen uit te lezen
         De functie: getStationPhaseCalibration
         elke .dat file bevat 96*512*2 doubles
         voor 96 rcus, 512 frequenties, een complexe waarde
         maar nu vraag ik me wel weer af of de frequenties of de rcus eerst komen
       */
      //NL stations: 768 kB, Int'l: 1.5 MB. Drop optional ASCI header. See also Station/StationCal/writeCalTable.m
      //itsDataset->dipoleCalibrationDelay().value = ???; // Pim can compute this from the GainCurve below
      //itsDataset->dipoleCalibrationDelayUnit().value = 's';
      //itsDataset->dipoleCalibrationGainCurve().create(???.size()).set(???); // st cal table
      //write cal tables into proper n-dimensional h5 data set, not attribute! Add access functions to DAL?

      // Skip if station is not participating in the observation (should not happen).
      if (stationMetaData.available && 2u * 3u * header.rcuID + 2u < stationMetaData.antPositions.size()) {
        /*TODO
         * Selecting the right positions depends on the antenna set. Checking vs the tables in
         * lhn001:/home/veen/lus/src/code/data/lofar/antennapositions/ can help, but their repos may be outdated.
         */
        vector<double> antPos(3);
        antPos[0] = stationMetaData.antPositions[2u * 3u * header.rcuID];
        antPos[1] = stationMetaData.antPositions[2u * 3u * header.rcuID + 1u];
        antPos[2] = stationMetaData.antPositions[2u * 3u * header.rcuID + 2u];
        itsDataset->antennaPosition().create(antPos.size()).set(antPos); // absolute position

        itsDataset->antennaPositionUnit().value = "m";
        itsDataset->antennaPositionFrame().value = parset.positionType(); // "ITRF"

        /*
         * The normal vector and rotation matrix are actually per antenna field,
         * but given the HBA0/HBA1 "ears" depending on antenna set, it was
         * decided to store them per antenna.
         */
        itsDataset->antennaNormalVector().create(stationMetaData.normalVector.size()).set(stationMetaData.normalVector);       // 3 doubles
        itsDataset->antennaRotationMatrix().create(stationMetaData.rotationMatrix.size()).set(stationMetaData.rotationMatrix); // 9 doubles, 3x3, row-major
      }

      // Tile beam is the analog beam. Only HBA can have one analog beam; optional.
      if (parset.haveAnaBeam()) {
        vector<double> anaBeamDir(parset.getAnaBeamDirection());
        itsDataset->tileBeam().create(anaBeamDir.size()).set(anaBeamDir); // always for beam 0
        itsDataset->tileBeamUnit().value = "m";
        itsDataset->tileBeamFrame().value = parset.getAnaBeamDirectionType(); // idem

        //itsDataset->tileBeamDipoles().create(???.size()).set(???);

        //itsDataset->tileCoefUnit().value = ???;
        //itsDataset->tileBeamCoefs().value = ???;

        // Relative position within the tile.
        //itsDataset->tileDipolePosition().value = ???;
        //itsDataset->tileDipolePositionUnit().value = ???;
        //itsDataset->tileDipolePositionFrame().value = ???;
      }

      itsDataset->dispersionMeasure().value = parset.dispersionMeasure(0, 0); // beam, pencil TODO: adapt too if >1 beam?
      itsDataset->dispersionMeasureUnit().value = "pc/cm^3";
    }

    bool TBB_Dipole::hasAllZeroDataSamples(const TBB_Payload& payload, size_t nTrSamples) const
    {
      /*
       * Good data only has a few consecutive zero values, so this loop terminates
       * quickly, unless the antenna is broken or disabled, which happens sometimes.
       */
      for (size_t i = 0; i < nTrSamples; i++) {
        if (payload.data[i] != 0) {
          return false;
        }
      }

      return true;
    }

//////////////////////////////////////////////////////////////////////////////

    TBB_Station::TBB_Station(const string& stationName, Mutex& h5Mutex, const Parset& parset,
                             const StationMetaData& stationMetaData, const string& h5Filename)
      : itsH5File(dal::TBB_File(h5Filename, dal::TBB_File::CREATE))
      , itsH5Mutex(h5Mutex)
      , itsStation(itsH5File.station(stationName))
      , itsDipoles(MAX_RSPBOARDS /* = per station*/ * NR_RCUS_PER_RSPBOARD) // = 192 for int'l stations
      , itsParset(parset)
      , itsStationMetaData(stationMetaData)
      , itsSubbandInfo(getSubbandInfo(parset))
      , itsH5Filename(h5Filename)
    {
      initCommonLofarAttributes();
      initTBB_RootAttributesAndGroups(stationName);
    }

    TBB_Station::~TBB_Station()
    {
      /*
       * Apart from the main thread, also potentially (rarely) executed by an output thread on failed
       * to insert new TBB_Station object into an std::map. For the output thread case, do dc and slH5.
       */
      ScopedDelayCancellation dc;
      try {
        ScopedLock slH5(itsH5Mutex);
        itsStation.nofDipoles().value = itsStation.dipoles().size();
      } catch (exception& exc) { // dal::DALException or worse
        LOG_WARN_STR("TBB: failed to set station NOF_DIPOLES attribute: " << exc.what());
      }
    }

    double TBB_Station::getSubbandCentralFreq(unsigned subbandNr, unsigned nyquistZone, double sampleFreq) const
    {
      return (nyquistZone - 1 + (double)subbandNr / RSP_NR_SUBBANDS) * sampleFreq / 2.0;
    }

    SubbandInfo TBB_Station::getSubbandInfo(const Parset& parset) const
    {
      SubbandInfo info;

      int operatingMode = itsParset.getInt("Observation.TBB.TBBsetting.operatingMode", 0);
      if (operatingMode == TBB_SPECTRAL_MODE) {
        vector<unsigned> tbbSubbandList(parset.getUint32Vector("Observation.TBB.TBBsetting.subbandList", true));
        if (tbbSubbandList.empty() || tbbSubbandList.size() > MAX_TBB_SPECTRAL_NSAMPLES) {
          throw CoInterfaceException("TBB: spectral mode selected, but empty or too long subband list provided");
        }
        sort(tbbSubbandList.begin(), tbbSubbandList.end());

        unsigned nyquistZone = parset.nyquistZone();
        unsigned sampleFreq = parset.clockSpeed() / 1000000;
        info.centralFreqs.reserve(tbbSubbandList.size());
        for (size_t i = 0; i < tbbSubbandList.size(); ++i) {
          info.centralFreqs.push_back(getSubbandCentralFreq(tbbSubbandList[i], nyquistZone, sampleFreq));
        }

        // "Invert" tbbSubbandList, such that we can later simply lookup where to store a subband.
        info.storageIndices.resize(RSP_NR_SUBBANDS, (unsigned)-1);
        for (unsigned i = 0; i < tbbSubbandList.size(); ++i) {
          unsigned sbNr = tbbSubbandList[i];
          if (sbNr >= RSP_NR_SUBBANDS) {
            throw CoInterfaceException("TBB: indicated subband number too high");
          }
          info.storageIndices[sbNr] = i;
        }
      }

      return info;
    }

    string TBB_Station::getRawFilename(unsigned rspID, unsigned rcuID) const
    {
      string rawFilename(itsH5Filename);
      string rsprcuStr(formatString("_%03u%03u", rspID, rcuID));
      size_t pos = rawFilename.find('_', rawFilename.find('_') + 1);
      rawFilename.insert(pos, rsprcuStr); // insert _rsp/rcu IDs after station name (2nd '_')
      rawFilename.resize(rawFilename.size() - (sizeof(".h5") - 1));
      rawFilename.append(".raw");
      return rawFilename;
    }

    void TBB_Station::processPayload(const TBB_Frame& frame)
    {
      // Guard against bogus incoming rsp/rcu IDs with at().
      TBB_Dipole& dipole(itsDipoles.at(frame.header.rspID * NR_RCUS_PER_RSPBOARD + frame.header.rcuID));

      // Each dipole stream is sent to a single port (thread), so no need to grab a mutex here to avoid double init.
      if (!dipole.isInitialized()) {
        string rawFilename(getRawFilename(frame.header.rspID, frame.header.rcuID));
        // Do pass a ref to the h5 mutex for when writing into the HDF5 file.
        dipole.init(frame.header, itsParset, itsStationMetaData, itsSubbandInfo,
                    rawFilename, itsStation, itsH5Mutex);
      }

      if (itsSubbandInfo.centralFreqs.empty()) { // transient mode
        dipole.processTransientFrameData(frame);
      } else { // spectral mode
        dipole.processSpectralFrameData(frame, itsSubbandInfo);
      }
    }

    string TBB_Station::utcTimeStr(double time) const
    {
      time_t timeSec = static_cast<time_t>(floor(time));
      unsigned long timeNSec = static_cast<unsigned long>(round( (time - floor(time)) * 1e9 ));

      char utc_str[50];
      struct tm tm;
      gmtime_r(&timeSec, &tm);
      if (strftime(utc_str, sizeof(utc_str), "%Y-%m-%dT%H:%M:%S", &tm) == 0) {
        return "";
      }

      return formatString("%s.%09luZ", utc_str, timeNSec);
    }

    double TBB_Station::toMJD(double time) const
    {
      // January 1st, 1970, 00:00:00 (GMT) equals 40587.0 Modify Julian Day number
      return 40587.0 + time / (24 * 60 * 60);
    }

    void TBB_Station::initCommonLofarAttributes()
    {
      itsH5File.groupType().value = "Root";

      //itsH5File.fileName() is set by DAL
      //itsH5File.fileDate() is set by DAL
      //itsH5File.fileType() is set by DAL
      //itsH5File.telescope() is set by DAL

      itsH5File.projectID().value = itsParset.getString("Observation.Campaign.name", "");
      itsH5File.projectTitle().value = itsParset.getString("Observation.Scheduler.taskName", "");
      itsH5File.projectPI().value = itsParset.getString("Observation.Campaign.PI", "");
      ostringstream oss;
      // Use ';' instead of ',' to pretty print, because ',' already occurs in names (e.g. Smith, J.).
      writeVector(oss, itsParset.getStringVector("Observation.Campaign.CO_I", ""), "; ", "", "");
      itsH5File.projectCOI().value = oss.str();
      itsH5File.projectContact().value = itsParset.getString("Observation.Campaign.contact", "");

      itsH5File.observationID().value = formatString("%u", itsParset.observationID());

      itsH5File.observationStartUTC().value = utcTimeStr(itsParset.startTime());
      itsH5File.observationStartMJD().value = toMJD(itsParset.startTime());

      // The stop time can be a bit further than the one actually specified, because we process in blocks.
      unsigned nrBlocks = floor((itsParset.stopTime() - itsParset.startTime()) / itsParset.CNintegrationTime()); // TODO: check vs bf: unsigned nrBlocks = parset.nrBeamFormedBlocks();
      double stopTime = itsParset.startTime() + nrBlocks * itsParset.CNintegrationTime();

      itsH5File.observationEndUTC().value = utcTimeStr(stopTime);
      itsH5File.observationEndMJD().value = toMJD(stopTime);

      itsH5File.observationNofStations().value = itsParset.nrStations(); // TODO: SS beamformer?
      // For the observation attribs, dump all stations participating in the observation (i.e. allStationNames(), not mergedStationNames()).
      // This may not correspond to which station HDF5 groups will be written for TBB, but that is true anyway, regardless of any merging.
      vector<string> allStNames(itsParset.allStationNames());
      itsH5File.observationStationsList().create(allStNames.size()).set(allStNames); // TODO: SS beamformer?

      double subbandBandwidth = itsParset.subbandBandwidth();
      double channelBandwidth = itsParset.channelWidth();

      // if PPF is used, the frequencies are shifted down by half a channel
      // We'll annotate channel 0 to be below channel 1, but in reality it will
      // contain frequencies from both the top and the bottom half-channel.
      double frequencyOffsetPPF = itsParset.nrChannelsPerSubband() > 1 ? 0.5 * channelBandwidth : 0.0;

      const vector<double> subbandCenterFrequencies(itsParset.subbandToFrequencyMapping());

      double min_centerfrequency = *min_element( subbandCenterFrequencies.begin(), subbandCenterFrequencies.end() );
      double max_centerfrequency = *max_element( subbandCenterFrequencies.begin(), subbandCenterFrequencies.end() );
      double sum_centerfrequencies = accumulate( subbandCenterFrequencies.begin(), subbandCenterFrequencies.end(), 0.0 );

      itsH5File.observationFrequencyMax().value = (max_centerfrequency + subbandBandwidth / 2 - frequencyOffsetPPF) / 1e6;
      itsH5File.observationFrequencyMin().value = (min_centerfrequency - subbandBandwidth / 2 - frequencyOffsetPPF) / 1e6;
      itsH5File.observationFrequencyCenter().value = (sum_centerfrequencies / subbandCenterFrequencies.size() - frequencyOffsetPPF) / 1e6;
      itsH5File.observationFrequencyUnit().value = "MHz";

      itsH5File.observationNofBitsPerSample().value = itsParset.nrBitsPerSample();
      itsH5File.clockFrequency().value = itsParset.clockSpeed() / 1e6;
      itsH5File.clockFrequencyUnit().value = "MHz";

      itsH5File.antennaSet().value = itsParset.antennaSet();
      itsH5File.filterSelection().value = itsParset.getString("Observation.bandFilter", "");

      unsigned nrSAPs = itsParset.nrBeams();
      vector<string> targets(nrSAPs);

      for (unsigned sap = 0; sap < nrSAPs; sap++) {
        targets[sap] = itsParset.beamTarget(sap);
      }

      itsH5File.targets().create(targets.size()).set(targets);

#ifndef TBB_WRITER_VERSION
      itsH5File.systemVersion().value = LOFAR::StorageVersion::getVersion();
#else
      itsH5File.systemVersion().value = TBB_WRITER_VERSION;
#endif

      //itsH5File.docName() is set by DAL
      //itsH5File.docVersion() is set by DAL

      itsH5File.notes().value = "";
    }

    // The writer creates one HDF5 file per station, so create only one Station Group here.
    void TBB_Station::initTBB_RootAttributesAndGroups(const string& stName)
    {
      int operatingMode = itsParset.getInt("Observation.TBB.TBBsetting.operatingMode", 0);
      if (operatingMode == TBB_SPECTRAL_MODE) {
        itsH5File.operatingMode().value = "spectral";
        itsH5File.spectralTransformSize().value = SPECTRAL_TRANSFORM_SIZE;
      } else {
        itsH5File.operatingMode().value = "transient";
      }

      itsH5File.nofStations().value = 1u;

      // Find the station name we are looking for and retrieve its pos using the found idx.
      vector<double> stPos;

      vector<string> obsStationNames(itsParset.allStationNames());
      vector<string>::const_iterator nameIt(obsStationNames.begin());

      vector<double> stationPositions(itsParset.positions()); // len must be (is generated as) 3x #stations
      vector<double>::const_iterator posIt(stationPositions.begin());
      string stFullName;
      for (; nameIt != obsStationNames.end(); ++nameIt, posIt += 3) {
        stFullName = *nameIt;
        if (stName == stFullName.substr(0, stName.size())) { // for TBB, consider "CS001" == "CS001HBA0" etc
          break;
        }
      }
      if (nameIt != obsStationNames.end() && posIt < stationPositions.end()) { // found?
        stPos.assign(posIt, posIt + 3);
      } else { // N/A, but create the group anyway to be able to store incoming data.
        stFullName.clear();
      }
      itsStation.create();
      initStationGroup(itsStation, stName, stFullName, stPos);

      // Trigger Group
      dal::TBB_Trigger tg(itsH5File.trigger());
      tg.create();
      initTriggerGroup(tg);
    }

    void TBB_Station::initStationGroup(dal::TBB_Station& st, const string& stName,
                                       const string& stFullName, const vector<double>& stPosition)
    {
      st.groupType().value = "StationGroup";
      st.stationName().value = stName;

      if (!stPosition.empty()) {
        st.stationPosition().create(stPosition.size()).set(stPosition);
        st.stationPositionUnit().value = "m";
        st.stationPositionFrame().value = itsParset.positionType();
      }

      // digital beam(s)
      if (itsParset.nrBeams() > 0) { // TODO: adapt DAL, so we can write all digital beams, analog too if tiles (HBA)
        vector<double> beamDir(itsParset.getBeamDirection(0));
        st.beamDirection().create(beamDir.size()).set(beamDir);
        st.beamDirectionUnit().value = "m";
        st.beamDirectionFrame().value = itsParset.getBeamDirectionType(0);
      }

      // Parset clockCorrectionTime() also returns 0.0 if stFullName is unknown. Avoid this ambiguity.
      try {
        double clockCorr = itsParset.getDouble(string("PIC.Core.") + stFullName + ".clockCorrectionTime");
        st.clockOffset().value = clockCorr;
        st.clockOffsetUnit().value = "s";
      } catch (APSException& exc) {
        LOG_WARN_STR("TBB: failed to write station clock offset and unit attributes: " << exc);
      }

      //st.nofDipoles.value is set at the end (destr)
    }

    void TBB_Station::initTriggerGroup(dal::TBB_Trigger& tg)
    {
      tg.groupType().value = "TriggerGroup";
      tg.triggerType().value = "Unknown";
      tg.triggerVersion().value = 0; // There is no trigger algorithm info available to us yet.

      // Trigger parameters (how to decide if there is a trigger; per obs)
      try {
        tg.paramCoincidenceChannels().value = itsParset.getInt   ("Observation.ObservationControl.StationControl.TBBControl.NoCoincChann");
        tg.paramCoincidenceTime().value = itsParset.getDouble("Observation.ObservationControl.StationControl.TBBControl.CoincidenceTime");
        tg.paramDirectionFit().value = itsParset.getString("Observation.ObservationControl.StationControl.TBBControl.DoDirectionFit");
        tg.paramElevationMin().value = itsParset.getDouble("Observation.ObservationControl.StationControl.TBBControl.MinElevation");
        tg.paramFitVarianceMax().value = itsParset.getDouble("Observation.ObservationControl.StationControl.TBBControl.MaxFitVariance");
      } catch (APSException& exc) {
        LOG_WARN_STR("TBB: Failed to write trigger parameters: " << exc);
      }

      // Trigger data (per trigger)
      // N/A atm

      /*
       * It is very likely that the remaining (optional) attributes and the trigger alg
       * will undergo many changes. TBB user/science applications will have to retrieve and
       * set the remaining fields "by hand" for a while using e.g. DAL by checking and
       * specifying each attribute name presumed available.
       * Until it is clear what is needed and available, this cannot be standardized.
       *
       * If you add fields using parset getTYPE(), catch the possible APSException as above.
       */

    }

//////////////////////////////////////////////////////////////////////////////

    TBB_StreamWriter::TBB_StreamWriter(TBB_Writer& writer, const string& inputStreamName,
                                       size_t expNTrSamples, const string& logPrefix,
                                       int& inExitStatus, int& outExitStatus)
      : itsWriter(writer)
      , itsInputStreamName(inputStreamName)
      , itsExpFrameSize(sizeof(TBB_Header) + expNTrSamples * sizeof(int16_t) + sizeof(uint32_t))
      , itsLogPrefix(logPrefix)
      , itsInExitStatus(inExitStatus)
      , itsOutExitStatus(outExitStatus)
    {
      itsFrameBuffers = new TBB_Frame[nrFrameBuffers];
      //itsReceiveQueue.reserve(nrFrameBuffers); // Queue does not support this...
      try {
        for (unsigned i = nrFrameBuffers; i > 0; ) {
          itsFreeQueue.append(&itsFrameBuffers[--i]);
        }
      } catch (exception& exc) {
        delete[] itsFrameBuffers;
        throw;
      }

      itsTimeoutStamp.tv_sec = 0;
      itsTimeoutStamp.tv_usec = 0;

      itsOutputThread = NULL;
      try {
        itsOutputThread = new Thread(this, &TBB_StreamWriter::mainOutputLoop, logPrefix + "OutputThread: ");
        itsInputThread = new Thread(this, &TBB_StreamWriter::mainInputLoop,  logPrefix + "InputThread: ");
      } catch (exception& exc) {
        if (itsOutputThread != NULL) {
          try {
            itsReceiveQueue.append(NULL); // tell output thread to stop
          } catch (exception& exc) {
            LOG_WARN_STR("TBB: failed to notify output thread to terminate: " << exc.what());
          }
          delete itsOutputThread;
        }
        delete[] itsFrameBuffers;
        throw;
      }

#ifdef DUMP_RAW_STATION_FRAMES
      struct timeval ts;
      ::gettimeofday(&ts, NULL);
      string rawStDataFilename("tbb_raw_station_frames_" + formatString("%ld_%p", ts.tv_sec, (void*)itsFrameBuffers) + ".fraw");
      try {
        itsRawStationData = new FileStream(rawStDataFilename, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
      } catch (exception& exc) {
        LOG_WARN_STR("Failed to open raw station data file: " << exc.what());
      }
#endif
    }

    TBB_StreamWriter::~TBB_StreamWriter()
    {
      // Only cancel the input thread, which will notify the output thread.
      itsInputThread->cancel();

#ifdef DUMP_RAW_STATION_FRAMES
      delete itsRawStationData;
#endif
      delete itsInputThread;
      delete itsOutputThread;
      delete[] itsFrameBuffers;
    }

    time_t TBB_StreamWriter::getTimeoutStampSec() const
    {
      return itsTimeoutStamp.tv_sec; // racy read (and no access once guarantee), but only to terminate after timeout
    }

    void TBB_StreamWriter::frameHeaderLittleToHost(TBB_Header& header) const
    {
      header.seqNr = le32toh(header.seqNr); // set to 0 for crc16, otherwise unused
      header.time = le32toh(header.time);
      header.sampleNr = le32toh(header.sampleNr);
      header.nOfSamplesPerFrame = le16toh(header.nOfSamplesPerFrame);
      header.nOfFreqBands = le16toh(header.nOfFreqBands);
      header.spare = le16toh(header.spare); // unused
      header.crc16 = le16toh(header.crc16);
    }

    void TBB_StreamWriter::correctSampleNr(TBB_Header& header) const
    {
      /*
       * LOFAR uses a sample rate of either 200 or 160 MHz.
       * In transient mode, at 200 MHz we get 1024 samples per frame, and thus 195213.5 frames per second.
       * This means that every 2 seconds, a frame overlaps a seconds boundary. But the sample values generated
       * by the RSPs start at zero for each second, even if it should start at 512 for odd timestamps at 200 MHz.
       * At 160 MHz sample rate, an integer number of frames fits in a second (156250), so no correction is needed.
       */
      if (header.sampleFreq == 200 && header.time & 1) {
        header.sampleNr += header.nOfSamplesPerFrame / 2;
      }
    }

    /*
     * Assumes that the seqNr field in the TBB_Frame at buf has been zeroed.
     * Takes a ptr to a complete header. (Drop too small frames earlier.)
     */
    bool TBB_StreamWriter::crc16tbb(const TBB_Header* header)
    {
      itsCrc16gen.reset();

      const char* ptr = reinterpret_cast<const char*>(header); // to char* for strict-aliasing
      for (unsigned i = 0; i < sizeof(*header) - sizeof(header->crc16); i += 2) {
        int16_t val;
        memcpy(&val, &ptr[i], sizeof val); // strict-aliasing safe
        val = __bswap_16(val);
        itsCrc16gen.process_bytes(&val, sizeof val);
      }

      // It is also possible to process header->crc16 and see if checksum() equals 0.
      uint16_t crc16val = header->crc16;
#if __BYTE_ORDER == __BIG_ENDIAN || defined WORDS_BIGENDIAN // for cross-compilation on little endian; fails for big->little
      crc16val = __bswap_16(crc16val);
#endif
      return itsCrc16gen.checksum() == crc16val;
    }

    /*
     * Note: The nTrSamples arg is without the space taken by the crc32 in payload (drop too small frames earlier)
     * and in terms of the transient sample size, i.e. sizeof(int16_t).
     */
    bool TBB_Dipole::crc32tbb(const TBB_Payload* payload, size_t nTrSamples)
    {
      itsCrc32gen.reset();

      const char* ptr = reinterpret_cast<const char*>(payload->data); // to char* for strict-aliasing
      for (unsigned i = 0; i < nTrSamples * sizeof(int16_t); i += 2) {
        int16_t val;
        memcpy(&val, &ptr[i], sizeof val); // strict-aliasing safe
        val = __bswap_16(val);
        itsCrc32gen.process_bytes(&val, sizeof val);
      }

      // It is also possible to process crc32val and see if checksum() equals 0.
      uint32_t crc32val;
      memcpy(&crc32val, &ptr[nTrSamples * sizeof(int16_t)], sizeof crc32val); // idem
#if __BYTE_ORDER == __BIG_ENDIAN || defined WORDS_BIGENDIAN // for cross-compilation on little endian; fails for big->little
      crc32val = __bswap_32(crc32val);
#endif
      return itsCrc32gen.checksum() == crc32val;
    }

    /*
     * Process the incoming TBB header.
     * Note that this function may update the header, but not its crc, so you cannot re-verify it.
     */
    void TBB_StreamWriter::processHeader(TBB_Header& header, size_t recvPayloadSize)
    {
      header.seqNr = 0; // For the header crc. Don't save/restore it as we don't need this field.
      if (!crc16tbb(&header)) {
        /*
         * The TBB spec states that each frame has the same fixed length, so the previous values are a good base guess if the header crc fails.
         * But it is not clear if it is worth the effort to try to guess to fix something up. For now, drop and log.
         */
        THROW(TBB_MalformedFrameException, "crc16: " << header); // header not yet bswapped on _big_ endian
      }

      /*
       * Use received size instead of received nOfSamplesPerFrame header field to access data, to be safe.
       * Just write it into the header; it's most likely already there.
       */
      if (recvPayloadSize < 2 * sizeof(int16_t) + sizeof(uint32_t)) {
        // Drop it. The data crc routine only works for at least 2 transient or 1 spectral sample(s) + a crc32.
        THROW(TBB_MalformedFrameException, "dropping too small frame: " << recvPayloadSize);
      }
      frameHeaderLittleToHost(header);
      // Verify indicated sample freq, also to reject zeroed headers, which the crc16tbb does not reject.
      if (header.sampleFreq != 200 && header.sampleFreq != 160) {
        THROW(TBB_MalformedFrameException, "dropping frame with invalid sample frequency in frame header: " << header.sampleFreq);
      }

      size_t sampleSize;
      if (header.nOfFreqBands == 0) { // transient mode TODO: do not rely on data to check data size!
        correctSampleNr(header);
        sampleSize = sizeof(int16_t);
      } else { // spectral mode
        sampleSize = 2 * sizeof(int16_t);
      }
      // Div with a bad recvPayloadSize could round. Causes crc32 error at worst, but avoids wrong or misaligned memory access.
      header.nOfSamplesPerFrame = (recvPayloadSize - sizeof(uint32_t)) / sampleSize;
    }

    void TBB_StreamWriter::mainInputLoop()
    {
      // Always (try to) notify output thread to stop at the end, else we may hang.
      class NotifyOutputThread
      {
        Queue<TBB_Frame*>& queue;
      public:
        NotifyOutputThread(Queue<TBB_Frame*>& queue) : queue(queue)
        {
        }
        ~NotifyOutputThread()
        {
          try {
            queue.append(NULL);
          } catch (exception& exc) {
            LOG_WARN_STR("TBB: may have failed to notify output thread to terminate: " << exc.what());
          }
        }
      } notifyOutThr(itsReceiveQueue);

      Stream* stream;
      try {
        stream = createStream(itsInputStreamName, true);
      } catch (Exception& exc) { // SystemCallException or CoInterfaceException (or TimeOutException)
        LOG_WARN_STR(itsLogPrefix << exc);
        itsInExitStatus = 1;
        return;
      }
      LOG_INFO_STR(itsLogPrefix << "reading incoming data from " << itsInputStreamName);

      while (1) {
        TBB_Frame* frame;

        try {
          frame = itsFreeQueue.remove();

          size_t nread = stream->tryRead(frame, itsExpFrameSize); // read() once for udp

          // Notify master that we are still busy. (Racy, but ok, see the timeoutstamp decl.)
          ::gettimeofday(&itsTimeoutStamp, NULL);

#ifdef DUMP_RAW_STATION_FRAMES
          try {
            itsRawStationData->write(frame, nread);
          } catch (exception& exc) { /* open() probably failed, don't spam */ }
#endif

          if (nread < sizeof(TBB_Header)) {
            throw TBB_MalformedFrameException("dropping too small frame");
          }
          processHeader(frame->header, nread - sizeof(TBB_Header));

          itsReceiveQueue.append(frame);

        } catch (TBB_MalformedFrameException& mffExc) {
          LOG_WARN_STR(itsLogPrefix << mffExc);
          try {
            itsFreeQueue.append(frame);
          } catch (exception& exc) {
            LOG_WARN_STR(itsLogPrefix << "may have lost a frame buffer (1): " << exc.what());
          }
        } catch (Stream::EndOfStreamException& ) { // after end of stream, for input from file or pipe
          break;
        } catch (exception& exc) {
          LOG_FATAL_STR(itsLogPrefix << exc.what());
          itsInExitStatus = 1;
          break;
        } catch (...) { // thread cancellation exc induced after timeout, for input from udp
          delete stream;
          throw; // mandatory
        }
      }

      delete stream;
    }

    void TBB_StreamWriter::mainOutputLoop()
    {
      bool running = true;
      while (running) {
        TBB_Frame* frame;
        try {
          frame = NULL;
          frame = itsReceiveQueue.remove();
          if (frame == NULL) {
            break;
          }

#ifdef PRINT_QUEUE_LEN
          LOG_INFO_STR(itsLogPrefix << "recvqsz=" << itsReceiveQueue.size());
#endif

          TBB_Station* station = itsWriter.getStation(frame->header);
          station->processPayload(*frame);

          // Tolerate the following exceptions. Maybe next rsp/rcu is ok; probably fatal too...
        } catch (SystemCallException& exc) {
          LOG_WARN_STR(itsLogPrefix << exc);
        } catch (StorageException& exc) {
          LOG_WARN_STR(itsLogPrefix << exc);
        } catch (dal::DALException& exc) {
          LOG_WARN_STR(itsLogPrefix << exc.what());
        } catch (out_of_range& exc) {
          LOG_WARN_STR(itsLogPrefix << exc.what());

          // Config/parset and other errors are fatal.
        } catch (exception& exc) {
          LOG_FATAL_STR(itsLogPrefix << exc.what());
          itsOutExitStatus = 1;
          running = false;
        }

        if (frame != NULL) {
          try {
            itsFreeQueue.append(frame);
          } catch (exception& exc) {
            LOG_WARN_STR(itsLogPrefix << "may have lost a frame buffer (2): " << exc.what());
          }
        }
      }
    }

//////////////////////////////////////////////////////////////////////////////

    TBB_Writer::TBB_Writer(const vector<string>& inputStreamNames, const Parset& parset,
                           const StationMetaDataMap& stationMetaDataMap,
                           const string& outDir, const string& logPrefix,
                           vector<int>& thrExitStatus)
      : itsParset(parset)
      , itsStationMetaDataMap(stationMetaDataMap)
      , itsOutDir(outDir)
      , itsRunNr(0)
    {
      // Mask all signals to inherit for workers. This forces signals to be delivered to the main thread.
      struct SigMask {
        sigset_t sigset_old;

        SigMask()
        {
          sigset_t sigset_all_masked;
          ::sigfillset(&sigset_all_masked);
          if (::pthread_sigmask(SIG_SETMASK, &sigset_all_masked, &sigset_old) != 0) {
            LOG_WARN_STR("TBB: pthread_sigmask() failed to mask signals to inherit for worker threads.");
          }
        }

        ~SigMask()
        {
          if (::pthread_sigmask(SIG_SETMASK, &sigset_old, NULL) != 0) {
            LOG_WARN_STR("TBB: pthread_sigmask() failed to restore signals. We may be deaf to signals.");
          }
        }
      } sigm;

      itsUnknownStationMetaData.available = false;

      size_t expNTrSamples; // in terms of the transient sample size
      int operatingMode = itsParset.getInt("Observation.TBB.TBBsetting.operatingMode", 0);
      if (operatingMode == TBB_TRANSIENT_MODE) {
        expNTrSamples = DEFAULT_TBB_TRANSIENT_NSAMPLES;
      } else if (operatingMode == TBB_SPECTRAL_MODE) {
        expNTrSamples = 2 * MAX_TBB_SPECTRAL_NSAMPLES;
      } else {
        expNTrSamples = DEFAULT_TBB_TRANSIENT_NSAMPLES;
        LOG_WARN("TBB: Failed to get operating mode from parset, assuming transient");
      }

      itsStreamWriters.reserve(inputStreamNames.size());
      for (unsigned i = 0; i < inputStreamNames.size(); i++) {
        itsStreamWriters.push_back(new TBB_StreamWriter(*this, inputStreamNames[i], expNTrSamples,
                                                        logPrefix, thrExitStatus[2 * i], thrExitStatus[2 * i + 1]));
      }
    }

    TBB_Writer::~TBB_Writer()
    {
      for (unsigned i = itsStreamWriters.size(); i > 0; ) {
        delete itsStreamWriters[--i];
      }

      map<unsigned, TBB_Station* >::iterator it(itsStations.begin());
      for (; it != itsStations.end(); ++it) {
        delete it->second;
      }
    }

    TBB_Station* TBB_Writer::getStation(const TBB_Header& header)
    {
      ScopedLock sl(itsStationsMutex); // protect against insert below
      map<unsigned, TBB_Station*>::iterator stIt(itsStations.find(header.stationID));
      if (stIt != itsStations.end()) {
        return stIt->second; // common case
      }

      // Create new station with HDF5 file and station HDF5 group.
      string stationName(dal::stationIDToName(header.stationID));
      string h5Filename(createNewTBB_H5Filename(header, stationName));
      StationMetaDataMap::const_iterator stMdIt(itsStationMetaDataMap.find(header.stationID));
      // If not found, station is not participating in the observation. Should not happen, but don't panic.
      const StationMetaData& stMetaData = stMdIt == itsStationMetaDataMap.end() ? itsUnknownStationMetaData : stMdIt->second;

      TBB_Station* station;
      {
        ScopedLock slH5(itsH5Mutex);
        station = new TBB_Station(stationName, itsH5Mutex, itsParset, stMetaData, h5Filename);
      }

      try {
        return itsStations.insert(make_pair(header.stationID, station)).first->second;
      } catch (exception& exc) {
        delete station;
        throw;
      }
    }

    string TBB_Writer::createNewTBB_H5Filename(const TBB_Header& header, const string& stationName)
    {
      const string typeExt("tbb.h5");
      string obsIDStr(formatString("%u", itsParset.observationID()));

      // Use the recording time of the first (received) frame as timestamp.
      struct timeval tv;
      tv.tv_sec = header.time;
      unsigned long usecNr;
      if (header.nOfFreqBands == 0) { // transient mode
        usecNr = header.sampleNr;
      } else { // spectral mode
        usecNr = header.bandSliceNr >> TBB_SLICE_NR_SHIFT;
      }
      tv.tv_usec = static_cast<unsigned long>(round( static_cast<double>(usecNr) / header.sampleFreq ));

      // Generate the output filename, because for TBB it is not in the parset.
      // From LOFAR-USG-ICD005 spec named "LOFAR Data Format ICD File Naming Conventions", by A. Alexov et al.
      const char output_format[] = "D%Y%m%dT%H%M"; // without secs
      const char output_format_secs[] = "%06.3fZ"; // total width of ss.sss is 6
      const char output_format_example[] = "DYYYYMMDDTHHMMSS.SSSZ";
      string triggerDateTime(formatFilenameTimestamp(tv, output_format, output_format_secs, sizeof(output_format_example)));
      string h5Filename(itsOutDir + "L" + obsIDStr + "_" + stationName + "_" + triggerDateTime + "_" + typeExt);

      // If the file already exists, add a run nr and retry. (might race and doesn't check .raw, but good enough)
      // If >1 stations per node, start at the prev run nr if any (hence itsRunNr).
      if (itsRunNr == 0) {
        if (::access(h5Filename.c_str(), F_OK) != 0 && errno == ENOENT) {
          // Does not exist (or broken dir after all, or dangling sym link...). Try this one.
          return h5Filename;
        } else {         // exists, inc run number
          itsRunNr = 1;
        }
      }

      size_t pos = h5Filename.size() - typeExt.size();
      string runNrStr(formatString("R%03u_", itsRunNr));
      h5Filename.insert(pos, runNrStr);
      while (itsRunNr < 1000 && ( ::access(h5Filename.c_str(), F_OK) == 0 || errno != ENOENT )) {
        itsRunNr += 1;
        runNrStr = formatString("R%03u_", itsRunNr);
        h5Filename.replace(pos, runNrStr.size(), runNrStr);
      }
      if (itsRunNr == 1000) { // run number is supposed to fit in 3 digits
        throw StorageException("failed to generate new .h5 filename after trying 1000 filenames.");
      }

      return h5Filename;
    }

    time_t TBB_Writer::getTimeoutStampSec(unsigned streamWriterNr) const
    {
      return itsStreamWriters[streamWriterNr]->getTimeoutStampSec();
    }

  } // namespace Cobalt
} // namespace LOFAR

#endif // HAVE_DAL


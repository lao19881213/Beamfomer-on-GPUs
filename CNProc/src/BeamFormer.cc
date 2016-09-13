//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <BeamFormer.h>

#include <Interface/MultiDimArray.h>
#include <Interface/Exceptions.h>
#include <Interface/SubbandMetaData.h>
#include <Common/Timer.h>
#include <Common/LofarLogger.h>
#include <cassert>
#include <algorithm>

#ifndef BEAMFORMER_C_IMPLEMENTATION
#include <BeamFormerAsm.h>
#endif

namespace LOFAR {
namespace RTCP {

static NSTimer formBeamsTimer("BeamFormer::formBeams()", true, true);
static NSTimer mergeStationsTimer("BeamFormer::mergeStations()", true, true);

BeamFormer::BeamFormer(const Parset &parset)
:
  itsDelays(parset.nrStations(), BEST_NRBEAMS),
  itsParset(parset),
  itsStationIndices(initStationIndices(parset)),
  itsNrStations(parset.nrStations()),
  itsValidStations(BEST_NRBEAMS),
  itsNrChannels(parset.nrChannelsPerSubband()),
  itsNrSamples(parset.CNintegrationSteps()),
  itsChannelBandwidth(parset.subbandBandwidth() / parset.CNintegrationSteps())
{
  initStationMergeMap(parset.tabList());
}

Matrix<std::vector<unsigned> > BeamFormer::initStationIndices(const Parset &parset)
{
  Matrix<std::vector<unsigned> > indexMatrix(parset.nrBeams(), parset.maxNrTABs());

  for (unsigned sap = 0; sap < parset.nrBeams(); sap++) {
    for (unsigned pencil = 0; pencil < parset.nrTABs(sap); pencil++) {
      const std::vector<std::string> stations = parset.TABStationList(sap, pencil);
      std::vector<unsigned> &indexList = indexMatrix[sap][pencil];

      indexList.resize(stations.size());

      for (unsigned s = 0; s < stations.size(); s++)
        indexList[s] = parset.stationIndex(stations[s]);

      std::sort(indexList.begin(), indexList.end());  
    }
  }

  return indexMatrix;
}

void BeamFormer::initStationMergeMap(const std::vector<unsigned> &station2BeamFormedStation)
{
  if (station2BeamFormedStation.empty()) {
    // beamforming disabled -- assignment is 1:1
    itsMergeSourceStations.resize(itsNrStations);
    itsMergeDestStations.resize(itsNrStations);

    for (unsigned i = 0; i<itsNrStations; i ++) {
      itsMergeSourceStations[i].push_back(i);
      itsMergeDestStations[i] = i;
    }
  } else {
    // beamforming enabled
    ASSERT(station2BeamFormedStation.size() == itsNrStations);

    unsigned nrMergedStations = *std::max_element(station2BeamFormedStation.begin(), station2BeamFormedStation.end()) + 1;

    itsMergeSourceStations.resize(nrMergedStations);
    itsMergeDestStations.resize(nrMergedStations);

    for (unsigned i = 0; i < itsNrStations; i ++) {
      unsigned id = station2BeamFormedStation[i];
      
      itsMergeSourceStations[id].push_back(i);
    }

    for (unsigned i = 0; i < nrMergedStations; i ++)
      itsMergeDestStations[i] = itsMergeSourceStations[i][0];
  }

  // reserve the same sizes for the vectors of valid stations
  itsValidMergeSourceStations.resize(itsMergeSourceStations.size());
  for (unsigned i = 0; i < itsValidMergeSourceStations.size(); i ++) {
    itsValidMergeSourceStations[i].reserve(itsMergeSourceStations[i].size());
  }
}


void BeamFormer::mergeStationFlags(const SampleData<> *in, SampleData<> *out)
{
  const unsigned upperBound = static_cast<unsigned>(itsNrSamples * itsNrChannels * BeamFormer::MAX_FLAGGED_PERCENTAGE);

  for (unsigned d = 0; d < itsMergeDestStations.size(); d ++) {
    unsigned			destStation	     = itsMergeDestStations[d];
    const std::vector<unsigned> &sourceStations      = itsMergeSourceStations[d];
    std::vector<unsigned>	&validSourceStations = itsValidMergeSourceStations[d];

    validSourceStations.clear();

    if (sourceStations.size() == 1) {
      // source and dest are the same (no beamforming), so checking for
      // MAX_FLAGGED_PERCENTAGE is unnecessary conservative
      validSourceStations.push_back(sourceStations[0]);
    } else {
      // copy valid stations from sourceStations -> validSourceStations
      for (unsigned s = 0; s < sourceStations.size(); s ++) {
        unsigned count = 0;
        for (unsigned ch = 0; ch < itsNrChannels; ch++) {
          count += in->flags[ch][sourceStations[s]].count();
        }

        if (count <= upperBound) {
          validSourceStations.push_back(sourceStations[s]);
        }
      }
    }   

    // conservative flagging: flag output if any input was flagged 
    if (validSourceStations.empty()) {
      // no valid stations: flag everything
      for (unsigned ch = 0; ch < itsNrChannels; ch++) {
        out->flags[ch][destStation].include(0, itsNrSamples);
      }
    } else {
      // some valid stations: merge flags

      if (validSourceStations[0] != destStation || in != out) {
        // dest station, which should be first in the list, was not valid
        for (unsigned ch = 0; ch < itsNrChannels; ch++) {
          out->flags[ch][destStation] = in->flags[ch][validSourceStations[0]];
        }
      }

      for (unsigned stat = 1; stat < validSourceStations.size(); stat ++) {
        for (unsigned ch = 0; ch < itsNrChannels; ch++) {
          out->flags[ch][destStation] |= in->flags[ch][validSourceStations[stat]];
        }
      }
    }
  }
}


void BeamFormer::computeFlags(const SampleData<> *in, SampleData<> *out, unsigned sap, unsigned firstBeam, unsigned nrBeams)
{
  const unsigned upperBound = static_cast<unsigned>(itsNrSamples * itsNrChannels * BeamFormer::MAX_FLAGGED_PERCENTAGE);

  // conservative flagging: flag output if any input was flagged 
  for (unsigned pencil = 0; pencil < nrBeams; pencil ++) {
    itsValidStations[pencil].clear();
    for (unsigned ch = 0; ch < itsNrChannels; ch++) {
      out->flags[pencil][ch].reset();
    }

    const std::vector<unsigned> &stations = itsStationIndices[sap][firstBeam + pencil];

    for (unsigned s = 0; s < stations.size(); s ++) {
      unsigned stat = stations[s];

      // determine which stations have too much flagged data
      unsigned count = 0;
      for (unsigned ch = 0; ch < itsNrChannels; ch++) {
        count += in->flags[ch][stat].count();
      }
      if (count <= upperBound) {
        itsValidStations[pencil].push_back(stat);
        for (unsigned ch = 0; ch < itsNrChannels; ch++) {
          out->flags[pencil][ch] |= in->flags[ch][stat];
        }
      }
    }  
  }
}

#ifdef BEAMFORMER_C_IMPLEMENTATION
void BeamFormer::mergeStations(const SampleData<> *in, SampleData<> *out)
{
  for (unsigned i = 0; i < itsValidMergeSourceStations.size(); i ++) {
    const unsigned destStation = itsMergeDestStations[i];
    const std::vector<unsigned> &validSourceStations  = itsValidMergeSourceStations[i];

    if (validSourceStations.empty())
      continue;

    if (validSourceStations.size() == 1 && validSourceStations[0] == destStation)
      continue;

    float factor = 1.0 / validSourceStations.size();

    for (unsigned ch = 0; ch < itsNrChannels; ch ++) {
      for (unsigned time = 0; time < itsNrSamples; time ++) {
        if (!out->flags[ch][destStation].test(time)) {
          for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++) {
            fcomplex &dest = out->samples[ch][destStation][time][pol];

            if (validSourceStations[0] != destStation) {
              // first station is somewhere else; copy it
              dest = in->samples[ch][0][time][pol];
            }

            // combine the stations
            for (unsigned stat = 1; stat < validSourceStations.size(); stat ++)
              dest += in->samples[ch][validSourceStations[stat]][time][pol];

            dest *= factor;
          }
	}
      }
    }
  }  
}

void BeamFormer::computeComplexVoltages(const SampleData<> *in, SampleData<> *out, double baseFrequency, unsigned nrBeams)
{
  for (unsigned ch = 0; ch < itsNrChannels; ch ++) {
    double frequency = baseFrequency + ch * itsChannelBandwidth;

  // construct the weights, with zeroes for unused data
  fcomplex weights[itsNrStations][nrBeams] __attribute__ ((aligned(128)));

  for (unsigned s = 0; s < itsNrStations; s ++) {
    for (unsigned b = 0; b < nrBeams; b ++)
      weights[s][b] = makefcomplex(0,0);

  for (unsigned b = 0; b < nrBeams; b ++) {
    if (itsValidStations[b].size() > 0) {
      double averagingSteps = 1.0 / itsValidStations[b].size();
      double factor = averagingSteps;

      for (unsigned s = 0; s < itsValidStations[b].size(); s++) {
        unsigned stat = itsValidStations[b][s];

        weights[stat][b] = phaseShift(frequency, itsDelays[stat][b]) * factor;
      }
    }  
  }

  for (unsigned beam = 0; beam < nrBeams; beam ++) {
      for (unsigned time = 0; time < itsNrSamples; time ++) {
        // PPF.cc sets flagged samples to 0, so we can always add them. Since flagged
        // samples are typically rare, it's faster to not test the flags of every
        // sample. This can be sped up by traversing the flags in groups.
        if (1 || !out->flags[beam][ch].test(time)) {
          for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++) {
            fcomplex &dest  = out->samples[beam][ch][time][pol];
	    double averagingSteps = 1.0 / itsValidStations[beam].size();
            float factor = averagingSteps;

            // combine the stations for this beam
            dest = makefcomplex(0, 0);

            std::vector<unsigned> &stations = itsValidStations[beam];

            for (unsigned s = 0; s < stations.size(); s ++) {
              unsigned stat = stations[s];
              fcomplex shift = weights[stat][beam];

              dest += in->samples[ch][stat][time][pol] * shift;
            }

            dest *= factor;
          }
        } else {
          // data is flagged
          for (unsigned pol = 0; pol < NR_POLARIZATIONS; pol ++) {
            out->samples[beam][ch][time][pol] = makefcomplex(0, 0);
  	  }
	}
      }
    }
  }
  }
}

#else // ASM implementation

// what we can process in one go
static const unsigned NRSTATIONS = 6;
static const unsigned NRBEAMS = 3;
#define BEAMFORMFUNC _beamform_up_to_6_stations_and_3_beams
#define ADDFUNC(nr)  _add_ ## nr ## _single_precision_vectors

// the number of samples to do in one go, such that the
// caches are optimally used. 
//
// TIMESTEPSIZE and itsNrSamples need to be a multiple of 16, as the assembly code requires it
static const unsigned TIMESTEPSIZE = 128;

// convertes from filtereddata to either filtereddata (mergeStations) or beamformeddata (formBeams)
inline void BeamFormer::addUnweighedStations(const SampleData<> *in, SampleData<> *out, const unsigned stationIndices[], unsigned nrStations, unsigned channel, unsigned beamIndex, unsigned timeOffset, unsigned timeLength, bool replace, bool outputHasChannelFirst, float weight)
{
  unsigned outDim1 = outputHasChannelFirst ? channel : beamIndex;
  unsigned outDim2 = outputHasChannelFirst ? beamIndex : channel;

  // central beam (#0) has no weights, we can simply add the stations
  switch(nrStations) {
    case 0:
    default:
      THROW(CNProcException,"Requested to add " << nrStations << " stations, but can only add 1-6.");
      break;

// possible inputs
#define OUTPUT		(reinterpret_cast<float*>(out->samples[outDim1][outDim2][timeOffset].origin()))
#define STATION(nr)	(reinterpret_cast<const float*>(in->samples[channel][stationIndices[nr]][timeOffset].origin()))

// shorthand for the add functions
#define ADDGENERIC(nr,...)	ADDFUNC(nr)(OUTPUT, __VA_ARGS__, timeLength * NR_POLARIZATIONS * 2) /* 2 is for real/imag */

// adds stations, and the subtotal if needed (if stat!=0)
#define ADD(nr,nrplusone,...)	do {							\
                            if (replace) {						\
                              ADDGENERIC(nr,__VA_ARGS__);				\
                            } else {			        			\
                              ADDGENERIC(nrplusone,OUTPUT,__VA_ARGS__);	        	\
                            }							        \
                          } while(0);

    case 1:
      ADD(1, 2, STATION(0));
      break;

    case 2:
      ADD(2, 3, STATION(0), STATION(1));
      break;

    case 3:
      ADD(3, 4, STATION(0), STATION(1), STATION(2));
      break;

    case 4:
      ADD(4, 5, STATION(0), STATION(1), STATION(2), STATION(3));
      break;

    case 5:
      ADD(5, 6, STATION(0), STATION(1), STATION(2), STATION(3), STATION(4));
      break;

    case 6:
      ADD(6, 7, STATION(0), STATION(1), STATION(2), STATION(3), STATION(4), STATION(5));
      break;
  }

  for (unsigned i = 0; i < timeLength; i ++) {
    for (unsigned p = 0; p < NR_POLARIZATIONS; p ++) {
      out->samples[outDim1][outDim2][timeOffset+i][p] *= weight;
    }
  }

}

void BeamFormer::mergeStations(const SampleData<> *in, SampleData<> *out)
{
  for (unsigned i = 0; i < itsValidMergeSourceStations.size(); i ++) {
    unsigned destStation = itsMergeDestStations[i];
    const std::vector<unsigned> &validSourceStations  = itsValidMergeSourceStations[i];

    if (validSourceStations.empty()) {
      continue;
    }

    if (validSourceStations.size() == 1 && validSourceStations[0] == destStation) {
      continue;
    }

    unsigned nrStations = validSourceStations.size();
    float factor = 1.0 / nrStations;
    bool destValid = validSourceStations[0] == destStation;

    // do the actual beamforming
    for (unsigned ch = 0; ch < itsNrChannels; ch ++) {
      unsigned processStations = NRSTATIONS;
      unsigned processTime = TIMESTEPSIZE;
      bool replaceDest = !destValid && in == out; // if true, ignore values at destStation

      // add everything to the first station in the list
      for (unsigned stat = replaceDest ? 0 : 1; stat < nrStations; stat += processStations) {
        processStations = std::min(NRSTATIONS, nrStations - stat);

        for (unsigned time = 0; time < itsNrSamples; time += processTime) {
          processTime = std::min(TIMESTEPSIZE, itsNrSamples - time);

          addUnweighedStations(in, out, &validSourceStations[stat], processStations, ch, destStation, time, processTime, replaceDest, true, factor);
        }

        replaceDest = false;
      }
    }
  }
}

void BeamFormer::computeComplexVoltages(const SampleData<> *in, SampleData<> *out, double baseFrequency, unsigned nrBeams)
{
  // This routine does the actual beam forming.
  //
  // It is optimised to form at most NRBEAMS beams. Every beam is formed out of a set of stations.
  //
  // Several special cases are dealt with:
  //  1) beams formed out of 0 stations (because all stations were flagged)
  //  2) beams formed out of 1 station with 0 delay (fly's eye)
  //
  // These special cases are 'peeled off' the set of beams to form. Since NRBEAMS == 3, this
  // leaves us with either a continuous subset ([0], [1], [2], [0,1], [1,2], [0,1,2])
  // or a single special case ([0,2]) of beams that still need to be formed after dealing
  // with the special cases.
  //
  // Because beam forming might not use all stations at all, we also keep track
  // of the first and the last station used for beam forming. Unused stations in between
  // get assigned a weight of 0, so optimal performance is only obtained if the set of
  // stations to add is continuous.

  ASSERT( nrBeams <= NRBEAMS ); // we'd run out of our structures otherwise
  ASSERT( NRBEAMS == 3 ); // we rely on this below for special cases

  // determine the set of beams to form
  bool beamForm[NRBEAMS] = { false, false, false };

  for (unsigned b = 0; b < nrBeams; b ++) {
    // special case: nothing to do (all stations are fully flagged, for instance)
    bool empty = itsValidStations[b].size() == 0;

    // special case: fly's eye: a pencil beam with zero delay and using only a single station
    bool flysEye = itsValidStations[b].size() == 1 && itsDelays[itsValidStations[b][0]][b] == 0.0;

    if (empty)
      flagBeam(out, b);
    else if (flysEye)
      computeFlysEye(in, out, b);
    else
      beamForm[b] = true;
  }

  // determine bounds on the stations to use and the beams to form
  unsigned nrBeamsToForm = 0;
  unsigned firstBeam = 0, lastBeam = 0;
  unsigned firstStation = 0, lastStation = 0;

  for (unsigned b = 0; b < nrBeams; b ++) {
    if (beamForm[b]) {
      // first and last station for this beam -- use the fact that itsValidStations is sorted
      unsigned fs = itsValidStations[b][0];
      unsigned ls = itsValidStations[b][itsValidStations[b].size()-1];

      if (nrBeamsToForm == 0) {
        firstBeam = b;
        lastBeam  = b;
        firstStation = fs;
        lastStation  = ls;
      } else {
        lastBeam = b;
        firstStation = std::min(firstStation, fs);
        lastStation  = std::max(lastStation,  ls);
      }

      nrBeamsToForm++;
    }
  }

  if (nrBeamsToForm == 0)
    return; // nothing (further) to do

  // construct the weights, with zeros for unused data
  fcomplex weights[lastStation + 1][nrBeamsToForm] __attribute__ ((aligned(128)));

  // do the actual beamforming
  for (unsigned ch = 0; ch < itsNrChannels; ch ++) {
    double frequency = baseFrequency + ch * itsChannelBandwidth;

    // Stations not in itsValidStations are either not used for beam forming
    // or have too much flagged samples. They will get a weight of 0.
    memset(&weights[0][0], 0, sizeof weights);

    // Set the weights we do have.
    for (unsigned b = 0, beamIndex = 0; b < nrBeams; b ++) {
      if (beamForm[b]) {
        double averagingSteps = 1.0 / itsValidStations[b].size();
        double factor = averagingSteps; // add multiplication factors as needed

        for (unsigned s = 0; s < itsValidStations[b].size(); s++) {
          unsigned stat = itsValidStations[b][s];

          weights[stat][beamIndex] = phaseShift(frequency, itsDelays[stat][b]) * factor;
        }

        beamIndex++;
      }  
    }

    unsigned processStations = NRSTATIONS;
    unsigned processTime = TIMESTEPSIZE;

    // Iterate over the same portions of the input data as many times as possible to 
    // fully exploit the caches.

    // 2 cases:
    //  - consecutive beams to form (form02 is false)
    //  - form beam 0 and 2 (form02 is true): use a larger stride to skip beam 1 in the output
    // form only beams 0 and 2?
    const bool form02 = beamForm[0] && !beamForm[1] && beamForm[2];

    // stride between beams in the output
    const unsigned out_stride = out->samples.strides()[0] * sizeof out->samples[0][0][0][0] * (form02 ? 2 : 1);

    // stride between stations in the input
    const unsigned in_stride  = in->samples[0].strides()[0] * sizeof in->samples[0][0][0][0];

    // stride between weight sets for different stations
    const unsigned weights_stride = (&weights[1][0] - &weights[0][0]) * sizeof weights[0][0];

    for (unsigned stat = firstStation; stat <= lastStation; stat += processStations) {
      processStations = std::min(NRSTATIONS, lastStation - stat + 1);

      for (unsigned time = 0; time < itsNrSamples; time += processTime) {
        processTime = std::min(TIMESTEPSIZE, itsNrSamples - time);

        // beam form
        BEAMFORMFUNC(
          out->samples[firstBeam][ch][time].origin(),
          out_stride,

          in->samples[ch][stat][time].origin(),
          in_stride,

          // weights are always consecutive and start at index 0
          &weights[stat][0],
          weights_stride,

          processTime,
          stat == 0,
          processStations,
          nrBeamsToForm
       );
      }
    }
  }
}

#endif

void BeamFormer::computeDelays(const SubbandMetaData *metaData, unsigned sap, unsigned firstBeam, unsigned nrBeams)
{
  // Calculate the delays for each station for this integration period.

  // We assume that the delay compensation has already occurred for the central beam. Also,
  // we use the same delay for all time samples. This could be interpolated for TIMESTEPSIZE
  // portions, as used in computeComplexVoltages.

  /*
  // no need to zero the data, because delays for unused stations won't be accessed

  for (unsigned stat = 0; stat < itsNrStations; stat ++)
    for (unsigned pencil = 0; pencil < nrBeams; pencil ++)
      itsDelays[stat][pencil] = 0.0f;
  */    

  for (unsigned pencil = 0; pencil < nrBeams; pencil ++) {
    unsigned pencilIndex = firstBeam + pencil;
    const std::vector<unsigned> &stationIndices = itsStationIndices[sap][pencilIndex];

    // adding no stations means adding all stations
    for (unsigned s = 0; s < stationIndices.size(); s++ ) {
      // if we need to add all stations, no lookups are necessary
      unsigned stat = stationIndices[s];

      // we already compensated for the delay for the first beam
      const SubbandMetaData::beamInfo &centralBeamInfo = metaData->beams(stat)[0];
      double compensatedDelay = (centralBeamInfo.delayAfterEnd + centralBeamInfo.delayAtBegin) * 0.5;

      const SubbandMetaData::beamInfo &beamInfo = metaData->beams(stat)[pencilIndex + 1];

      // subtract the delay that was already compensated for
      itsDelays[stat][pencil] = (beamInfo.delayAfterEnd + beamInfo.delayAtBegin) * 0.5 - compensatedDelay;
    }  
  }
}

void BeamFormer::flagBeam(SampleData<> *out, unsigned beam) {
  for (unsigned ch = 0; ch < itsNrChannels; ch++) {
    out->flags[beam][ch].include(0, itsNrSamples);
  }
  memset(out->samples[beam].origin(), 0, out->samples[beam].num_elements() * sizeof out->samples[0][0][0][0]);
}


void BeamFormer::computeFlysEye(const SampleData<> *in, SampleData<> *out, unsigned beam) {
  unsigned src = itsValidStations[beam][0];

  // copy station src to dest
  for (unsigned ch = 0; ch < itsNrChannels; ch++) {
    out->flags[beam][ch] = in->flags[ch][src];
  }

  for (unsigned ch = 0; ch < itsNrChannels; ch ++)
    memcpy(out->samples[beam][ch].origin(),
            in->samples[ch][src].origin(), 
            in->samples[ch].strides()[0] * sizeof in->samples[0][0][0][0]);
}

void BeamFormer::mergeStations(SampleData<> *sampleData)
{
  ASSERT(sampleData->samples.shape()[0] == itsNrChannels);
  ASSERT(sampleData->samples.shape()[1] == itsNrStations);
  ASSERT(sampleData->samples.shape()[2] >= itsNrSamples);
  ASSERT(sampleData->samples.shape()[3] == NR_POLARIZATIONS);

  mergeStationsTimer.start();
  mergeStationFlags(sampleData, sampleData);
  mergeStations(sampleData, sampleData);
  mergeStationsTimer.stop();
}

void BeamFormer::formBeams(const SubbandMetaData *metaData, SampleData<> *sampleData, BeamFormedData *beamFormedData, unsigned subband, unsigned sap, unsigned firstBeam, unsigned nrBeams)
{
  ASSERT(sampleData->samples.shape()[0] == itsNrChannels);
  ASSERT(sampleData->samples.shape()[1] == itsNrStations);
  ASSERT(sampleData->samples.shape()[2] >= itsNrSamples);
  ASSERT(sampleData->samples.shape()[3] == NR_POLARIZATIONS);

  ASSERT(nrBeams > 0);
  ASSERT(nrBeams <= BEST_NRBEAMS);

#if !defined BEAMFORMER_C_IMPLEMENTATION
  ASSERT(TIMESTEPSIZE % 16 == 0);

  if (itsNrSamples % 16 > 0) {
    // for asm routines
    THROW(CNProcException, "nrSamplesPerIntegration (" << itsNrSamples << ") needs to be a multiple of 16");
  }
#endif

  // TODO: fetch a list of stations to beam form. for now, we assume
  // we use all stations
  //

  // take the 2nd PPF into account, which shifts the center down frequency by half
  // a channel.
  double baseFrequency = itsParset.channel0Frequency(subband);

  formBeamsTimer.start();

  // perform beam forming
  computeDelays(metaData, sap, firstBeam, nrBeams);
  computeFlags(sampleData, beamFormedData, sap, firstBeam, nrBeams);
  computeComplexVoltages(sampleData, beamFormedData, baseFrequency, nrBeams);

  formBeamsTimer.stop();
}

void BeamFormer::preTransposeBeam(const BeamFormedData *in, PreTransposeBeamFormedData *out, unsigned inbeam)
{ 
  // split polarisations and real/imaginary part of beams
  ASSERT(in->samples.shape()[0] > inbeam);
  ASSERT(in->samples.shape()[1] == itsNrChannels);
  ASSERT(in->samples.shape()[2] >= itsNrSamples);
  ASSERT(in->samples.shape()[3] == NR_POLARIZATIONS);

  ASSERT(out->samples.shape()[0] == NR_POLARIZATIONS * 2);
  ASSERT(out->samples.shape()[1] == itsNrChannels);
  ASSERT(out->samples.shape()[2] >= itsNrSamples);

  ASSERT(NR_POLARIZATIONS == 2);

  for (unsigned c = 0; c < itsNrChannels; c ++) {
    out->flags[c] = in->flags[inbeam][c];
  }

#if 0
  /* reference implementation */
  for (unsigned c = 0; c < itsNrChannels; c ++)
    for (unsigned t = 0; t < itsNrSamples; t ++) {
      out->samples[0][c][t] = real(in->samples[inbeam][c][t][0]);
      out->samples[1][c][t] = imag(in->samples[inbeam][c][t][0]);
      out->samples[2][c][t] = real(in->samples[inbeam][c][t][1]);
      out->samples[3][c][t] = imag(in->samples[inbeam][c][t][1]);
    }    
#else
  /* in_stride == 1 */
  /* out_stride == 1 */

  for (unsigned c = 0; c < itsNrChannels; c ++) {
    const fcomplex *inb = &in->samples[inbeam][c][0][0];
    float *outbXr, *outbXi, *outbYr, *outbYi;

    outbXr = &out->samples[0][c][0];
    outbXi = &out->samples[1][c][0];
    outbYr = &out->samples[2][c][0];
    outbYi = &out->samples[3][c][0];

    for (unsigned s = 0; s < itsNrSamples; s ++) {
      *outbXr = real(*inb);
      *outbXi = imag(*inb);
      inb++;

      *outbYr = real(*inb);
      *outbYi = imag(*inb);
      inb++;

      outbXr ++;
      outbXi ++;
      outbYr ++;
      outbYi ++;
    }
  }
#endif  
}


void BeamFormer::postTransposeBeam(const TransposedBeamFormedData *in, FinalBeamFormedData *out, unsigned sb, unsigned nrChannels, unsigned nrSamples)
{
  ASSERT(in->samples.shape()[0] > sb);
  ASSERT(in->samples.shape()[1] == nrChannels);
  ASSERT(in->samples.shape()[2] >= nrSamples);

  ASSERT(out->samples.shape()[0] >= nrSamples);
  ASSERT(out->samples.shape()[1] > sb);
  ASSERT(out->samples.shape()[2] == nrChannels);

  out->flags[sb] = in->flags[sb];

#if defined USE_VALGRIND // TODO: if "| 2" is removed, this should not be necessary anymore
  for (unsigned t = nrSamples; t < out->samples.shape()[0]; t ++)
    for (unsigned c = 0; c < nrChannels; c ++)
      out->samples[t][sb][c] = 0;
#endif

#if 1
  /* reference implementation */
  for (unsigned t = 0; t < nrSamples; t ++)
    for (unsigned c = 0; c < nrChannels; c ++)
      out->samples[t][sb][c] = in->samples[sb][c][t];
#else
#endif
}

} // namespace RTCP
} // namespace LOFAR



#ifndef LOFAR_CNPROC_BEAMFORMER_H
#define LOFAR_CNPROC_BEAMFORMER_H

#include <vector>
#include <cmath>

#include <Interface/Parset.h>
#include <Interface/StreamableData.h>
#include <Interface/BeamFormedData.h>
#include <Interface/SubbandMetaData.h>
#include <BandPass.h>
#include <CN_Math.h>

#if 0 || !defined HAVE_BGP
#define BEAMFORMER_C_IMPLEMENTATION
#endif

namespace LOFAR {
namespace RTCP {

/*

   This beam former supports three modes:

   1) merging stations, using the mergeStations function.
   2) creating tied-array beams (TABs), using the formBeams function.
   3) creating a 'fly's eye', which is a variation on mode 2, except
      that each station creates its own beam (i.e. the data is copied).

   Merging stations
   -------------------------

   Stations are merged in-place according to the station2BeamFormedStation array, which is a mapping
   source -> dest of length nrStations. Multiple sources with the same dest are added and stored at dest.
   If the station2BeamFormedStation array is empty, source and dest are mapped 1:1 and no stations are merged.

   Creating tied-array beams
   -------------------------

   Pencil beams are created by specifying their number as nrBeams in the formBeams function, and by the
   delays as provided by the metaData given to formBeams. If nrTABs = 0, the target data structure
   remains untouched.

*/

class BeamFormer
{
  public:
    static const float MAX_FLAGGED_PERCENTAGE = 1.0f;

    // ideal number of beams that can be calculated in one go
    static const unsigned BEST_NRBEAMS = 3;

    BeamFormer(const Parset &parset);

    // merges stations into superstations in sampleData
    void mergeStations( SampleData<> *sampleData );

    // fills beamFormedData with pencil beams
    void formBeams( const SubbandMetaData *metaData, SampleData<> *sampleData, BeamFormedData *beamFormedData, unsigned subband, unsigned sap, unsigned firstBeam, unsigned nrBeams );

    // rearrange dimensions in preparation for transpose
    void preTransposeBeam( const BeamFormedData *in, PreTransposeBeamFormedData *out, unsigned inbeam );

    // rearrange dimensions into final order after transpose
    void postTransposeBeam( const TransposedBeamFormedData *in, FinalBeamFormedData *out, unsigned sb, unsigned nrChannels, unsigned nrSamples );

    // return the station mapping
    std::vector<unsigned> &getStationMapping();

    Matrix<double>          itsDelays; // [itsNrStations][BEST_NRBEAMS]
  private:
    unsigned calcNrBeamFormedStations();
    Matrix<std::vector<unsigned> > initStationIndices( const Parset &parset );
    void initStationMergeMap( const std::vector<unsigned> &station2BeamFormedStation );

    // extracts the delays from the metaData, and transforms them if necessary
    void computeDelays( const SubbandMetaData *metaData, unsigned sap, unsigned firstBeam, unsigned nrBeams );

    dcomplex phaseShift( const double frequency, const double delay ) const;

    void addUnweighedStations( const SampleData<> *in, SampleData<> *out, const unsigned stationIndices[], unsigned nrStations, unsigned channel, unsigned beamIndex, unsigned timeOffset, unsigned timeLength, bool first, bool outputHasChannelFirst, float weight );

    // sets the flags in beamFormedData, and decides which stations should be added
    void computeFlags( const SampleData<> *sampleData, SampleData<> *beamFormedData, unsigned sap, unsigned firstBeam, unsigned nrBeams );
    void mergeStationFlags( const SampleData<> *in, SampleData<> *out );

    // the actual beam former
    void mergeStations( const SampleData<> *in, SampleData<> *out );
    void computeComplexVoltages( const SampleData<> *in, SampleData<> *out, double baseFrequency, unsigned nrBeams );
    void flagBeam(SampleData<> *out, unsigned beam);
    void computeFlysEye(const SampleData<> *in, SampleData<> *out, unsigned beam);

    const Parset            &itsParset;

    const Matrix<std::vector<unsigned> > itsStationIndices;
    const unsigned          itsNrStations;

    Vector<std::vector<unsigned> > itsValidStations;
    const unsigned          itsNrChannels;
    const unsigned          itsNrSamples;
    const double            itsChannelBandwidth;

    // a station is 'valid' if the samples do not contain too much flagged data. invalid stations
    // are ignored by the beamformer.

    std::vector<std::vector<unsigned> > itsMergeSourceStations;        // [i] = [a,b,c] => beam i is a+b+c
    std::vector<unsigned>               itsMergeDestStations;          // [i] = a => beam i is stored at a
    std::vector<std::vector<unsigned> > itsValidMergeSourceStations;   // subset of itsMergeSourceStations,
                                                                       // containing only valid stations
};

inline dcomplex BeamFormer::phaseShift( const double frequency, const double delay ) const
{
  const double phaseShift = delay * frequency;
  const double phi = -2 * M_PI * phaseShift;

  return cosisin(phi);
}

inline std::vector<unsigned> &BeamFormer::getStationMapping() { 
  return itsMergeDestStations;
}


} // namespace RTCP
} // namespace LOFAR

#endif

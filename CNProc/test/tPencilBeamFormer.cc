#include <lofar_config.h>

#include <BeamFormer.h>
#include <Common/lofar_complex.h>
#include <Interface/FilteredData.h>
#include <Interface/BeamFormedData.h>
#include <vector>
#include <boost/format.hpp>

using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace LOFAR::TYPES;
using boost::format;

#define NRSTATIONS              12
#define NRPENCILBEAMS           12

#define NRCHANNELS              64
#define NRSAMPLES               1024 // keep computation time short, 128 is minimum (see BeamFormer.cc)
#define NRSUBBANDS              3

#define CENTERFREQUENCY         (80.0e6)
#define BASEFREQUENCY           (CENTERFREQUENCY - (NRCHANNELS/2)*CHANNELBW)
#define CHANNELBW               (1.0*200e6/1024/NRCHANNELS)

#define TOLERANCE               1e-6

inline dcomplex phaseShift( const double frequency, const double delay )
{
  const double phaseShift = delay * frequency;
  const double phi = -2 * M_PI * phaseShift;
  return cosisin(phi);
}

inline bool same( const float a, const float b )
{
  return abs(a-b) < TOLERANCE;
}

Parset createParset()
{
  string stationNames = "[";
  for(unsigned i = 0; i < NRSTATIONS; i++) {
    if(i>0) stationNames += ", ";

    stationNames += str(format("CS%03u") % i);
  }
  stationNames += "]";

  Parset p;
  p.add("Observation.bandFilter",               "LBA_30_70");
  p.add("Observation.channelsPerSubband",       str(format("%u") % NRCHANNELS));
  p.add("OLAP.CNProc.integrationSteps",         str(format("%u") % NRSAMPLES));
  p.add("Observation.sampleClock",              "200");
  p.add("OLAP.storageStationNames",             stationNames);
  p.add("Observation.subbandList",              str(format("[%u*100]") % NRSUBBANDS));
  p.add("Observation.beamList",                 str(format("[%u*0]") % NRSUBBANDS));
  p.add("OLAP.tiedArrayStationNames",           "[]");
  p.add("OLAP.CNProc.tabList",                  "[]");
  p.add("Observation.Beam[0].nrTiedArrayBeams", str(format("%u") % NRPENCILBEAMS));

  for(unsigned i = 0; i < NRPENCILBEAMS; i++) {
    p.add(str(format("Observation.Beam[0].tiedArrayBeam[%u].angle1") % i), "0.0");
    p.add(str(format("Observation.Beam[0].tiedArrayBeam[%u].angle2") % i), "0.0");
    p.add(str(format("Observation.Beam[0].tiedArrayBeam[%u].stationList") % i), "[]");
  }  

  return p;
}

SubbandMetaData createSubbandMetaData( const Parset &p )
{
  (void)p;

  SubbandMetaData metaData(NRSTATIONS, NRPENCILBEAMS);

  for (unsigned i = 0; i < NRSTATIONS; i++) {
    metaData.alignmentShift(i) = 0;

    metaData.beams(i)->delayAtBegin = 0.0;
    metaData.beams(i)->delayAfterEnd = 0.0;
  }  

  return metaData;
}

void test_flyseye() {
  std::vector<unsigned> stationMapping(0);
  FilteredData   in( NRSTATIONS, NRCHANNELS, NRSAMPLES );
  BeamFormedData out( NRPENCILBEAMS, NRCHANNELS, NRSAMPLES );

  assert( NRSTATIONS == NRPENCILBEAMS );

  // fill filtered data
  for( unsigned c = 0; c < NRCHANNELS; c++ ) {
    for( unsigned s = 0; s < NRSTATIONS; s++ ) {
      for( unsigned i = 0; i < NRSAMPLES; i++ ) {
        for( unsigned p = 0; p < NR_POLARIZATIONS; p++ ) {
          in.samples[c][s][i][p] = makefcomplex( s+1, 0 );
        }
      }
    }
  }

  // form beams
  Parset p = createParset();
  BeamFormer f = BeamFormer(p);
  SubbandMetaData m = createSubbandMetaData(p);
  f.mergeStations( &in );

  for( unsigned b = 0; b < NRPENCILBEAMS; b += BeamFormer::BEST_NRBEAMS ) {
    unsigned nrBeams = b + BeamFormer::BEST_NRBEAMS >= NRPENCILBEAMS
      ? NRPENCILBEAMS - b
      : BeamFormer::BEST_NRBEAMS;

    f.formBeams( &m, &in, &out, 0, 0, b, nrBeams );
  }

  // check beamformed data
  for( unsigned c = 0; c < NRCHANNELS; c++ ) {
    for( unsigned i = 0; i < NRSAMPLES; i++ ) {
      for( unsigned p = 0; p < NR_POLARIZATIONS; p++ ) {
        for( unsigned s = 0; s < NRSTATIONS; s++ ) {
          const unsigned b = s;

          assert( out.samples[b][c][i][p] == in.samples[c][s][i][p] );
        }
      }
    }
  }
}

void test_stationmerger() {
  std::vector<unsigned> stationMapping(3);
  FilteredData		in(NRSTATIONS, NRCHANNELS, NRSAMPLES);

  // fill filtered data
  for( unsigned c = 0; c < NRCHANNELS; c++ ) {
    for( unsigned s = 0; s < NRSTATIONS; s++ ) {
      for( unsigned i = 0; i < NRSAMPLES; i++ ) {
        for( unsigned p = 0; p < NR_POLARIZATIONS; p++ ) {
          in.samples[c][s][i][p] = makefcomplex( s+1, 0 );
        }
      }
    }
  }

  // create mapping
  stationMapping[0] = 0;
  stationMapping[1] = 1;
  stationMapping[2] = 1;

  // form beams
  Parset p = createParset();
  BeamFormer f = BeamFormer(p);
  f.mergeStations( &in );

  // check merged data
  for( unsigned c = 0; c < NRCHANNELS; c++ ) {
    for( unsigned i = 0; i < NRSAMPLES; i++ ) {
      for( unsigned p = 0; p < NR_POLARIZATIONS; p++ ) {
        fcomplex sums[NRSTATIONS];
        unsigned nrstations[NRSTATIONS];

        for( unsigned s = 0; s < NRSTATIONS; s++ ) {
          sums[s] = makefcomplex(s+1,0);
          nrstations[s] = 1;
        }
        for( unsigned s = 0; s < NRSTATIONS; s++ ) {
          if( stationMapping[s] != s ) {
            sums[stationMapping[s]] += makefcomplex( s+1, 0 );
            nrstations[stationMapping[s]]++;
          }
        }
        for( unsigned s = 0; s < NRSTATIONS; s++ ) {
          sums[s] /= nrstations[s];
        }

        for( unsigned s = 0; s < NRSTATIONS; s++ ) {
          if( !same(real(sums[s]),real(in.samples[c][s][i][p])) 
           || !same(imag(sums[s]),imag(in.samples[c][s][i][p])) ) {
            std::cerr << in.samples[c][s][i][p] << " =/= " << sums[s] << " for station " << s << " channel " << c << " sample " << i << " pol " << p << std::endl;
            exit(1);
          }
        }
      }
    }
  }
}

void test_beamformer() {
  std::vector<unsigned> stationMapping(0);
  FilteredData    in( NRSTATIONS, NRCHANNELS, NRSAMPLES );
  BeamFormedData  out( NRPENCILBEAMS, NRCHANNELS, NRSAMPLES );
  SubbandMetaData meta( NRSTATIONS, NRPENCILBEAMS );

  // fill filtered data
  for( unsigned c = 0; c < NRCHANNELS; c++ ) {
    for( unsigned s = 0; s < NRSTATIONS; s++ ) {
      for( unsigned i = 0; i < NRSAMPLES; i++ ) {
        for( unsigned p = 0; p < NR_POLARIZATIONS; p++ ) {
          in.samples[c][s][i][p] = makefcomplex( s+1, 0 );
        }
      }
    }
  }

  // fill weights
  for( unsigned s = 0; s < NRSTATIONS; s++ ) {
    meta.beams(s)[0].delayAtBegin = 
    meta.beams(s)[0].delayAfterEnd = 0.0;

    for( unsigned b = 1; b < NRPENCILBEAMS; b++ ) {
      meta.beams(s)[b].delayAtBegin = 
      meta.beams(s)[b].delayAfterEnd = 1.0 * s / b;
    }
  }

  // form beams
  Parset p = createParset();
  BeamFormer f = BeamFormer(p);

  f.mergeStations( &in );

  for( unsigned b = 0; b < NRPENCILBEAMS; b += 3 ) {
    unsigned nrBeams = b + 3 >= NRPENCILBEAMS ? NRPENCILBEAMS - b : 3;

    f.formBeams( &meta, &in, &out, 0, 0, b, nrBeams );
  }
/*
  // check beamformed data
  for( unsigned s = 0; s < NRSTATIONS; s++ ) {
    for( unsigned b = 0; b < NRPENCILBEAMS; b++ ) {
      assert( same( f.itsDelays[s][b], meta.beams(s)[b].delayAtBegin ) );
    }
  }

  const float averagingFactor = 1.0 / NRSTATIONS;
  const float factor = averagingFactor;

  for( unsigned b = 0; b < NRPENCILBEAMS; b++ ) {
    for( unsigned c = 0; c < NRCHANNELS; c++ ) {
      const double frequency = BASEFREQUENCY + c * CHANNELBW;

      for( unsigned i = 0; i < NRSAMPLES; i++ ) {
        assert( !out.flags[b].test(i) );

        for( unsigned p = 0; p < NR_POLARIZATIONS; p++ ) {
          fcomplex sum = makefcomplex( 0, 0 );

          for( unsigned s = 0; s < NRSTATIONS; s++ ) {
            dcomplex shift = phaseShift( frequency, meta.beams(s)[b].delayAtBegin );
            const fcomplex weight = makefcomplex(shift);

            sum += in.samples[c][s][i][p] * weight;
          }

          sum *= factor;

          if( !same(real(sum),real(out.samples[b][c][i][p])) 
           || !same(imag(sum),imag(out.samples[b][c][i][p])) ) {
            std::cerr << out.samples[b][c][i][p] << " =/= " << sum << " for beam " << b << " channel " << c << " sample " << i << " pol " << p << std::endl;
            exit(1);
          }
        }
      }
    }
  }
  */
}

void test_pretranspose()
{
  BeamFormedData in( 1, NRCHANNELS, NRSAMPLES );
  PreTransposeBeamFormedData out( 4, NRCHANNELS, NRSAMPLES );
  Parset p = createParset();
  BeamFormer f = BeamFormer(p);

  // fill input data
  for( unsigned c = 0; c < NRCHANNELS; c++ ) {
    for( unsigned i = 0; i < NRSAMPLES; i++ ) {
      real(in.samples[0][c][i][0]) = 1.0f * (c + i * NRCHANNELS + 1);
      imag(in.samples[0][c][i][0]) = 3.0f * (c + i * NRCHANNELS + 1);
      real(in.samples[0][c][i][1]) = 5.0f * (c + i * NRCHANNELS + 1);
      imag(in.samples[0][c][i][1]) = 7.0f * (c + i * NRCHANNELS + 1);
    }
  }

  f.preTransposeBeam( &in, &out, 0 );

  for( unsigned st = 0; st < 4; st++ ) {
    for( unsigned c = 0; c < NRCHANNELS; c++ ) {
      for( unsigned i = 0; i < NRSAMPLES; i++ ) {
        float &x = out.samples[st][c][i];
        float y;

        switch(st) {
          case 0:
            y = real(in.samples[0][c][i][0]);
            break;

          case 1:
            y = imag(in.samples[0][c][i][0]);
            break;

          case 2:
            y = real(in.samples[0][c][i][1]);
            break;

          case 3:
            y = imag(in.samples[0][c][i][1]);
            break;
        }

        if( !same(x, y) ) {
          std::cerr << "preTransposeBeams: Sample doesn't match for stokes #" << st << " channel " << c << " sample " << i << std::endl;
          exit(1);
        }
      }
    }
  }
}

void test_posttranspose()
{
  TransposedBeamFormedData in( NRSUBBANDS, NRCHANNELS, NRSAMPLES );
  FinalBeamFormedData out( NRSAMPLES, NRSUBBANDS, NRCHANNELS );
  Parset p = createParset();
  BeamFormer f = BeamFormer(p);

  // fill input data
  for( unsigned sb = 0; sb < NRSUBBANDS; sb++ ) {
    for( unsigned c = 0; c < NRCHANNELS; c++ ) {
      for( unsigned i = 0; i < NRSAMPLES; i++ ) {
        in.samples[sb][c][i] = 1.0f * (sb + c * NRSUBBANDS + i * NRSUBBANDS * NRCHANNELS +1);
      }
    }

    f.postTransposeBeam( &in, &out, sb, NRCHANNELS, NRSAMPLES );
  }  

  for( unsigned sb = 0; sb < NRSUBBANDS; sb++ ) {
    for( unsigned c = 0; c < NRCHANNELS; c++ ) {
      for( unsigned i = 0; i < NRSAMPLES; i++ ) {
        float &x = out.samples[i][sb][c];

        if( !same(x, in.samples[sb][c][i]) ) {
          std::cerr << "postTransposeBeams: Sample doesn't match for subband " << sb << " channel " << c << " sample " << i << std::endl;
          exit(1);
        }
      }
    }
  }

}

int main() {
  //test_flyseye();
  //test_stationmerger();
  test_beamformer();
  test_pretranspose();
  test_posttranspose();

  return 0;
}

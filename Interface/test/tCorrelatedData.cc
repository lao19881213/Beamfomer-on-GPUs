#include <lofar_config.h>

#include <Common/Timer.h>

#include <Interface/CorrelatedData.h>

#include <cassert>
#include <iostream>


using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace std;

int main(void)
{
  NSTimer timer("addition", true, false);

  unsigned nr_maxsamples[] = { 255, 65535, 1000000 }; // encode using 1, 2, 4 bytes, respectively
  unsigned nr_channels[]  = { 1, 16, 64, 256 };
  unsigned nr_stations[] = { 1, 2, 3, 4, 5, 24 };

  for( unsigned s = 0; s < sizeof nr_maxsamples / sizeof nr_maxsamples[0]; ++s )
    for( unsigned ch = 0; ch < sizeof nr_channels / sizeof nr_channels[0]; ++ch )
      for( unsigned st = 0; st < sizeof nr_stations / sizeof nr_stations[0]; ++st ) {
        unsigned ns = nr_maxsamples[s];
        unsigned nch = nr_channels[ch];
        unsigned nst = nr_stations[st];
        unsigned nbl = nst * (nst + 1) / 2;

        cout << nst << " stations (= " << nbl << " baselines), " << nch << " channels, " << ns << " samples" << endl;

        // we will test whether data1 + data2 = data3
        CorrelatedData data1(nst, nch, ns), data2(nst, nch, ns), data3(nst, nch, ns);

        // initialise data
        cout << "init" << endl;
        unsigned n = 0;

        for( unsigned i = 0; i < nbl; i++ ) {
          for( unsigned j = 0; j < nch; j++ ) {
            n++;
            
            data1.setNrValidSamples(i, j, (n*1) % (ns/2));
            data2.setNrValidSamples(i, j, (n*2) % (ns/2));
            data3.setNrValidSamples(i, j, ((n*1) % (ns/2)) + ((n*2) % (ns/2)));

            for( unsigned k = 0; k < 2; k++ ) {
              for( unsigned l = 0; l < 2; l++ ) {
                data1.visibilities[i][j][k][l] =    1 * ((i + j) * 10 + k * 2 + l);
                data2.visibilities[i][j][k][l] = 1000 * ((i + j) * 10 + k * 2 + l);
                data3.visibilities[i][j][k][l] = 1001 * ((i + j) * 10 + k * 2 + l);
              }
            }
          }
        }

        // add
        cout << "add" << endl;
        timer.start();
        data1 += data2;
        timer.stop();

        // verify
        cout << "verify" << endl;
        for( unsigned i = 0; i < nbl; i++ ) {
          for( unsigned j = 0; j < nch; j++ ) {
            //cout << data1.nrValidSamples(i, j) << " == " << data3.nrValidSamples(i, j) << endl;
            assert(data1.nrValidSamples(i, j) == data3.nrValidSamples(i, j));

            for( unsigned k = 0; k < 2; k++ ) {
              for( unsigned l = 0; l < 2; l++ ) {
                assert(
                  data1.visibilities[i][j][k][l] ==
                  data3.visibilities[i][j][k][l]
                  );
              }
            }
          }
        }

        cout << "ok" << endl;
      }

  return 0;
}

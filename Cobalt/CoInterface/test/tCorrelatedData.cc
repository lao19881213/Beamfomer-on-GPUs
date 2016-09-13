//# tCorrelatedData.cc
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
//# $Id: tCorrelatedData.cc 24553 2013-04-09 14:21:56Z mol $

#include <lofar_config.h>

#include <CoInterface/CorrelatedData.h>

#include <cassert>
#include <iostream>

#include <Common/Timer.h>


using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;

int main(void)
{
  NSTimer timer("addition", true, false);

  unsigned nr_maxsamples[] = { 255, 65535, 1000000 }; // encode using 1, 2, 4 bytes, respectively
  unsigned nr_channels[] = { 1, 16, 64, 256 };
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

            data1.setNrValidSamples(i, j, (n * 1) % (ns / 2));
            data2.setNrValidSamples(i, j, (n * 2) % (ns / 2));
            data3.setNrValidSamples(i, j, ((n * 1) % (ns / 2)) + ((n * 2) % (ns / 2)));

            for( unsigned k = 0; k < 2; k++ ) {
              for( unsigned l = 0; l < 2; l++ ) {
                data1.visibilities[i][j][k][l] = 1 * ((i + j) * 10 + k * 2 + l);
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
            assert(data1.getNrValidSamples(i, j) == data3.getNrValidSamples(i, j));

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


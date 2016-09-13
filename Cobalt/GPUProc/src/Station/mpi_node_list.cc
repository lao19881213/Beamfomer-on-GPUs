//# mpi_node_list.cc: Generate the MPI node list to execute over, from a parset.
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
//# $Id: mpi_node_list.cc 25888 2013-08-01 09:05:27Z mol $

#include <lofar_config.h>

#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <boost/format.hpp>

#include <Common/StreamUtil.h>
#include <CoInterface/Parset.h>

using namespace LOFAR;
using namespace LOFAR::Cobalt;
using namespace std;
using boost::format;

void usage(char **argv)
{
  cerr << "usage: " << argv[0] << " [-n] parset" << endl;
  cerr << endl;
  cerr << "-n: print node list" << endl;
  cerr << endl;
}

void print_node_list(Parset &ps)
{
  // Collect all host names
  vector<string> nodes;
  for (size_t node = 0; node < ps.settings.nodes.size(); ++node)
    nodes.push_back(ps.settings.nodes[node].hostName);

  // Output all host names
  writeVector(cout, nodes, ",", "", "");
  cout << endl;
}

int main(int argc, char **argv)
{
  INIT_LOGGER("mpi_node_list");

  bool nodelist = false;

  // parse all command-line options
  int opt;
  while ((opt = getopt(argc, argv, "n")) != -1) {
    switch (opt) {
    case 'n':
      nodelist = true;
      break;

    default: /* '?' */
      usage(argv);
      exit(1);
    }
  }

  // we expect a parset filename as an additional parameter
  if (optind >= argc) {
    usage(argv);
    exit(1);
  }

  // Create a parameters set object based on the inputs
  Parset ps(argv[optind]);

  // Perform the requested actions
  if (nodelist)
    print_node_list(ps);

  return 0;
}


//# tTBB_StaticMapping.cc
//#
//# Copyright (C) 2012
//# ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: tTBB_StaticMapping.cc 2465 2012-02-28 14:24:54Z amesfoort $

#include <lofar_config.h>

#include <cstdlib>
#include <iostream>

#include <Common/Exceptions.h>
#include <Storage/TBB_StaticMapping.h>

using namespace std;
using namespace LOFAR;

int main(int argc, char *argv[]) {
	// Locate TBB connection mapping file.
	// Select from either: argv[1], $LOFARROOT/etc/StaticMetaData/TBBConnections.dat, or ./TBBConnections.dat
	string tbbMappingFilename;
	if (argc > 1) {
		tbbMappingFilename = argv[1];
	} else {
		const string defaultTbbMappingFilename("TBBConnections.dat");
		char* lrpath = getenv("LOFARROOT");
		if (lrpath != NULL) {
			tbbMappingFilename = string(lrpath) + "/etc/StaticMetaData/";
		}
		tbbMappingFilename.append(defaultTbbMappingFilename);
	}

	try {
		// Open and read in.
		TBB_StaticMapping tsm(tbbMappingFilename);

		if (tsm.empty()) {
			throw Exception("Opened tbb static mapping file, but list of station names is empty");
		}

		// Show all.
		cout << "Found " << tsm.size() << " nodes with the following station and board names:" << endl;
		for (multimap<string, pair<string, string> >::const_iterator it(tsm.begin()); it != tsm.end(); ++it) {
			cout << "node: " << (*it).first << " -> (" << (*it).second.first << ", " << (*it).second.second << ")" << endl;
		}
	
		// Select all station or board names mapped to a given node.
		const string nodeName("locus029");

		vector<string> stations(tsm.getStationNames(nodeName));
		cout << nodeName << ": ";
		for (unsigned i = 0; i < stations.size(); i++) {
			cout << stations[i] << " ";
		}
		cout << endl;

		vector<string> boards(tsm.getBoardNames(nodeName));
		cout << nodeName << ": ";
		for (unsigned i = 0; i < boards.size(); i++) {
			cout << boards[i] << " ";
		}
		cout << endl;

	} catch (Exception& exc) {
		cerr << exc.what() << endl;
		return 1;
	}

	return 0;
}


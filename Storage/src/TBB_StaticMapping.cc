//# TBB_StaticMapping.cc: read out the TBB static station-node mapping
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
//# $Id: TBB_StaticMapping.cc 3375 2012-03-12 20:01:54Z amesfoort $

#include <lofar_config.h>

#include <Storage/TBB_StaticMapping.h>
#include <Common/StringUtil.h>
#include <Common/Exceptions.h>

#include <fstream>

using namespace std;

namespace LOFAR {

TBB_StaticMapping::TBB_StaticMapping() {
}

TBB_StaticMapping::TBB_StaticMapping(const string& filename) {
	parseStaticMapping(filename);
}

void TBB_StaticMapping::parseStaticMapping(const string& filename) {
	char buf[parseBufSize];
	const string ws(" \t");
	ifstream ifile(filename.c_str());
	if (!ifile) {
		throw IOException("Failed to open TBB static meta data file with station to node mapping");
	}

	while (ifile.getline(buf, parseBufSize).good()) {
		string sbuf(buf);

		size_t pos = sbuf.find('#'); // line comments
		sbuf = sbuf.substr(0, pos);
		vector<string> tokens(StringUtil::tokenize(sbuf, ws));

		// We expect 3 tokens (columns): stationName (0), board (1), destNode (2); ignore other tokens and "empty" lines.
		if (tokens.size() >= 3) {
			itsMapping.insert(make_pair(tokens[2], make_pair(tokens[0], tokens[1])));
		}
	}
}

multimap<string, pair<string, string> >::const_iterator TBB_StaticMapping::begin() const {
	return itsMapping.begin();
}

multimap<string, pair<string, string> >::const_iterator TBB_StaticMapping::end() const {
	return itsMapping.end();
}

size_t TBB_StaticMapping::size() const {
	return itsMapping.size();
}

bool TBB_StaticMapping::empty() const {
	return itsMapping.empty();
}

vector<string> TBB_StaticMapping::getStationNames(const string& nodeName) const {
	vector<string> mapping;

	for (pair<multimap<string, pair<string, string> >::const_iterator,
              multimap<string, pair<string, string> >::const_iterator> iters(
              itsMapping.equal_range(nodeName));
			iters.first != iters.second; ++iters.first) {
		mapping.push_back((*iters.first).second.first);
	}

	return mapping;
}

vector<string> TBB_StaticMapping::getBoardNames(const string& nodeName) const {
	vector<string> mapping;

	for (pair<multimap<string, pair<string, string> >::const_iterator,
              multimap<string, pair<string, string> >::const_iterator> iters(
              itsMapping.equal_range(nodeName));
			iters.first != iters.second; ++iters.first) {
		mapping.push_back((*iters.first).second.second);
	}

	return mapping;
}

} // ns LOFAR

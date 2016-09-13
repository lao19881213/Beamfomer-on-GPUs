//# TBB_StaticMapping.h: TBB static station-node mapping class
//# Can be used to parse LOFAR/MAC/Deployment/data/StaticMetaData/TBBConnections.dat
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
//# $Id: TBB_StaticMapping.h 2278 2012-03-12 20:01:54Z amesfoort $

#ifndef TBB_STATIC_MAPPING_H
#define TBB_STATIC_MAPPING_H 1

#include <string>
#include <vector>
#include <map>

namespace LOFAR {

class TBB_StaticMapping {
public:
	TBB_StaticMapping();

	// These two throw a LOFAR::IOException if filename could not be opened.
	explicit TBB_StaticMapping(const std::string& filename);
	void parseStaticMapping(const std::string& filename);

	std::multimap<std::string, std::pair<std::string, std::string> >::const_iterator begin() const;
	std::multimap<std::string, std::pair<std::string, std::string> >::const_iterator end() const;
	size_t size() const;
	bool empty() const;
	std::vector<std::string> getStationNames(const std::string& nodeName) const;
	std::vector<std::string> getBoardNames(const std::string& nodeName) const;

private:
	// Max line len in file is now 52, but need a bit more if >1 stations per dest node.
	static const size_t parseBufSize = 256;

	// Maps from node name to (station name, board name).
	std::multimap<std::string, std::pair<std::string, std::string> > itsMapping;
};


} // ns LOFAR

#endif // TBB_CONNECTIONS_H

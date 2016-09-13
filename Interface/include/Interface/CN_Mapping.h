//#  SparseSet.h: portable <bitset> adaptation
//#
//#  Copyright (C) 2006
//#  ASTRON (Netherlands Foundation for Research in Astronomy)
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: CN_Mapping.h 11930 2008-10-16 12:44:30Z gels $


#ifndef LOFAR_INTERFACE_CN_MAPPING_H
#define LOFAR_INTERFACE_CN_MAPPING_H


namespace LOFAR {
namespace RTCP {

class CN_Mapping
{
  public:
    // Reshuffle cores within different psets differently, to make the transpose
    // over the 3D-torus much more efficient.  Without reshuffling, transposing
    // cores often communicate in the same line or plane in the torus, causing
    // severe bottlenecks over a few links.  With reshuffling, there are more
    // redundant links, significantly improving the bandwidth.  TODO: improve
    // the reshuffling function further, to minimize transpose times.

    static unsigned mapCoreOnPset(unsigned core, unsigned pset);
    static unsigned reverseMapCoreOnPset(unsigned core, unsigned pset);
};

} // namespace RTCP
} // namespace LOFAR

#endif

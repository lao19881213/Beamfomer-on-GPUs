//#  GlobalVars.cc: global variables
//#
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
//#  $Id: ION_main.cc 15296 2010-03-24 10:19:41Z romein $

#include <lofar_config.h>

#include <GlobalVars.h>
#include <Interface/CN_Mapping.h>
#include <Interface/SmartPtr.h>
#include <Interface/Stream.h>
#include <boost/multi_array.hpp>

#if defined HAVE_FCNP && defined __PPC__ && !defined USE_VALGRIND
#include <FCNP/fcnp_ion.h>
#include <FCNP_ServerStream.h>
#endif

namespace LOFAR {
namespace RTCP {

unsigned				  myPsetNumber, nrPsets, nrCNcoresInPset;
std::vector<SmartPtr<Stream> >		  allIONstreams;
Matrix<SmartPtr<Stream> >		  allCNstreams;

std::vector<SmartPtr<StreamMultiplexer> > allIONstreamMultiplexers;

const char                       *cnStreamType;

Stream *createCNstream(unsigned pset, unsigned core, unsigned channel)
{
  // translate logical to physical core number
  core = CN_Mapping::mapCoreOnPset(core, myPsetNumber);

#if defined HAVE_FCNP && defined __PPC__ && !defined USE_VALGRIND
  ASSERT( pset == myPsetNumber );

  if (strcmp(cnStreamType, "FCNP") == 0)
    return new FCNP_ServerStream(core, channel);
#endif

  string descriptor = getStreamDescriptorBetweenIONandCN(cnStreamType, myPsetNumber, pset, core, nrPsets, nrCNcoresInPset, channel);

  return createStream(descriptor, true);
}


} // namespace RTCP
} // namespace LOFAR


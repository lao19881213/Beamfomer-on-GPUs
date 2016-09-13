//# tDAL: Test HDF5 routines through DAL
//#
//#  Copyright (C) 2011
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
//#  $Id: $

#include <lofar_config.h>

#ifdef HAVE_DAL

#include <dal/dal_version.h>
#include <iostream>
#include <string>

using namespace std;
using namespace dal;

int main() {
  if (!check_hdf5_versions()) {
    cerr << "HDF5 version mismatch. DAL was compiled with " << version_hdf5_headers_dal() << ", our headers are " << version_hdf5_headers_current() << ", our library is " << version_hdf5() << endl;
    return 1;
  }
  
  return 0;
}

#else

int main() {
  return 0;
}
#endif

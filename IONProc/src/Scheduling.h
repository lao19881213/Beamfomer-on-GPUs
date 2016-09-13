//#  OutputThread.h
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
//#  $Id: Scheduling.h 13340 2009-05-28 08:47:40Z mol $

#ifndef LOFAR_IONPROC_SCHEDULING_H
#define LOFAR_IONPROC_SCHEDULING_H

//# Never #include <config.h> or #include <lofar_config.h> in a header file!

namespace LOFAR {
namespace RTCP {

#if defined HAVE_BGP_ION
// Core 0 handles all ethernet and tree interrupts.  Do not run time-critical
// threads on this core.
extern void doNotRunOnCore0();
extern void runOnCore0();

// set thread priority. 0 = normal, 1 - 99 = real time
extern void setPriority(unsigned priority);
#endif

} // namespace RTCP
} // namespace LOFAR

#endif

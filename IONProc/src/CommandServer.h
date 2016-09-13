//#  CommandServer.h
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


//# Never #include <config.h> or #include <lofar_config.h> in a header file!

#if !defined LOFAR_RTCP_COMMAND_SERVER_H
#define LOFAR_RTCP_COMMAND_SERVER_H

#include <string>

#include <Common/Thread/Semaphore.h>
#include <Common/Thread/Thread.h>
#include <Interface/SmartPtr.h>


namespace LOFAR {
namespace RTCP {

class CommandServer
{
  public:
	      CommandServer();
	      ~CommandServer();

    void      start();

  private:
    void      commandMaster(), commandSlave();
    void      handleCommand(const std::string &);

    void      jobCleanUpThread();

    bool      itsQuit;
    Semaphore itsNrJobsCreated;
    SmartPtr<Thread> itsJobCleanUpThread;
};

} // namespace RTCP
} // namespace LOFAR

#endif

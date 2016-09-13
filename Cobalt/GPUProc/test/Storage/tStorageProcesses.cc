//# tStorageProcesses.cc
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: tStorageProcesses.cc 26812 2013-10-01 19:51:36Z mol $

#include <lofar_config.h>

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include <string>

#include <boost/format.hpp>

#include <Common/LofarLogger.h>
#include <CoInterface/Parset.h>

#include <GPUProc/Storage/StorageProcesses.h>
#include <GPUProc/Storage/SSH.h>

char pubkey[1024];
char privkey[1024];

using namespace LOFAR;
using namespace Cobalt;
using namespace std;

void test_simple()
{
  // Test whether executing an application works. The
  // communication protocol after startup is ignored.

  const char *USER = getenv("USER");
  Parset p;

  p.add("Observation.ObsID",               "12345");
  p.add("Cobalt.OutputProc.userName",      USER);
  p.add("Cobalt.OutputProc.sshPublicKey",  pubkey);
  p.add("Cobalt.OutputProc.sshPrivateKey", privkey);
  p.add("Cobalt.OutputProc.executable",    "/bin/echo");

  p.add("Observation.nrBeams",             "1");
  p.add("Observation.Beam[0].subbandList", "[0]");
  p.add("Observation.DataProducts.Output_Correlated.enabled", "true");
  p.add("Observation.DataProducts.Output_Correlated.filenames", "[SB0.MS]");
  p.add("Observation.DataProducts.Output_Correlated.locations", "[localhost:.]");
  p.updateSettings();

  {
    StorageProcesses sp(p, "");

    // give 'ssh localhost' time to succeed
    sleep(2);
  }
}

void test_protocol()
{
  // Test whether we follow the communication protocol
  // as expected by Storage_main.

  const char *USER = getenv("USER");
  Parset p;

  char cwd[1024];

  if (getcwd(cwd, sizeof cwd) == 0)
    THROW_SYSCALL("getcwd");

  p.add("Observation.ObsID",                 "12345");
  p.add("Cobalt.OutputProc.userName",        USER);
  p.add("Cobalt.OutputProc.sshPublicKey",    pubkey);
  p.add("Cobalt.OutputProc.sshPrivateKey",   privkey);
  p.add("Cobalt.OutputProc.executable",      str(boost::format("%s/DummyStorage") % cwd));
  p.add("Cobalt.FinalMetaDataGatherer.host",          "localhost");
  p.add("Cobalt.FinalMetaDataGatherer.userName",      USER);
  p.add("Cobalt.FinalMetaDataGatherer.sshPublicKey",  pubkey);
  p.add("Cobalt.FinalMetaDataGatherer.sshPrivateKey", privkey);

  p.add("Observation.nrBeams",             "1");
  p.add("Observation.Beam[0].subbandList", "[0]");
  p.add("Observation.DataProducts.Output_Correlated.enabled", "true");
  p.add("Observation.DataProducts.Output_Correlated.filenames", "[SB0.MS]");
  p.add("Observation.DataProducts.Output_Correlated.locations", "[localhost:.]");

  // DummyStorage already emulates the FinalMetaDataGatherer, so use a dummy
  // instead.
  p.add("Cobalt.FinalMetaDataGatherer.executable",      "/bin/echo");
  p.updateSettings();

  {
    StorageProcesses sp(p, "");

    // Give Storage time to log its parset
    sleep(2);

    // Give 10 seconds to exchange final meta data
    sp.forwardFinalMetaData(time(0) + 10);

    // Give 10 seconds to wrap up
    sp.stop(time(0) + 10);

    // Obtain LTA feedback
    ParameterSet feedbackLTA(sp.feedbackLTA());

    ASSERT(feedbackLTA.getString("foo","") == "bar");
  }
}

int main()
{
  INIT_LOGGER( "tStorageProcesses" );

  // prevent stalls
  alarm(60);

  SSH_Init();

  if (!discover_ssh_keys(pubkey, sizeof pubkey, privkey, sizeof privkey))
    return 3;

  test_simple();
  test_protocol();

  SSH_Finalize();

  return 0;
}


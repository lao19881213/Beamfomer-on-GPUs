//# tSSH.cc
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
//# $Id: tSSH.cc 25779 2013-07-26 11:00:04Z mol $

#include <lofar_config.h>

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include <boost/format.hpp>

#include <Common/LofarLogger.h>
#include <Stream/SocketStream.h>

#include <GPUProc/Storage/SSH.h>

char pubkey[1024];
char privkey[1024];

using namespace LOFAR;
using namespace Cobalt;
using boost::format;


void test_SSHconnection( const string &stdout_str, const string &stderr_str, bool capture )
{
  const char *USER = getenv("USER");
  stringstream ssh_stdout, ssh_stderr;
  const string cmd = str(format("echo %s; echo %s 1>&2") % stdout_str % stderr_str);
  SSHconnection ssh("", "localhost", cmd, USER, pubkey, privkey, capture, ssh_stdout, ssh_stderr);

  ssh.start();

  struct timespec ts;
  ts.tv_sec = time(0) + 10;
  ts.tv_nsec = 0;

  ssh.wait(ts);

  // SSH (and bashrc) can insert headers and footers on stdout and stderr, so we need to
  // search for our echoed strings.

  if (capture)
    ASSERT(ssh.stdoutBuffer().find(stdout_str) != string::npos);
  else
    ASSERT(ssh_stdout.str().find(stdout_str) != string::npos);

  ASSERT(ssh_stderr.str().find(stderr_str) != string::npos);
}

int main()
{
  INIT_LOGGER( "tSSH" );

  SSH_Init();

  // discover a working private key
  if (!discover_ssh_keys(pubkey, sizeof pubkey, privkey, sizeof privkey))
    return 3;

  test_SSHconnection( "stdout read", "", false );
  test_SSHconnection( "",            "stderr read", false );
  test_SSHconnection( "stdout read", "stderr read", false );

  test_SSHconnection( "stdout capture", "", true );

  SSH_Finalize();

  return 0;
}


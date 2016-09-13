#include <lofar_config.h>

#include <SSH.h>
#include <unistd.h>
#include <time.h>
#include <cstdlib>
#include <cstdio>
#include <Stream/SocketStream.h>
#include <Common/LofarLogger.h>

// some useful environment variables
char *USER;
char *HOME;

// the existence of $HOME/.ssh/id_rsa is assumed,
// as well as the fact that it can be used to
// authenticate on localhost.
char privkey[1024];

using namespace LOFAR;
using namespace RTCP;


void test_SSHconnection( const char *cmd, bool capture ) {
#ifdef HAVE_LIBSSH2
  SSHconnection ssh("", "localhost", cmd, USER, privkey, capture);

  ssh.start();

  struct timespec ts;
  ts.tv_sec = time(0) + 10;
  ts.tv_nsec = 0;

  ssh.wait(ts);

  if (capture)
    cout << "Captured [" << ssh.stdoutBuffer() << "]" << endl;
#endif
}

int main() {
  INIT_LOGGER( "tSSH" );

  USER = getenv("USER");
  HOME = getenv("HOME");
  snprintf(privkey, sizeof privkey, "%s/.ssh/id_rsa", HOME);

  // can we even ssh to localhost?
  char sshcmd[1024];
  snprintf(sshcmd, sizeof sshcmd, "ssh %s@localhost -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o NoHostAuthenticationForLocalhost=yes -i %s echo system success", USER, privkey);
  int ret = system(sshcmd);
  if (ret < 0 || WEXITSTATUS(ret) != 0) {
    // no -- mark this test as unrunnable and don't attempt to try with libssh then
    return 3;
  }  

  SSH_Init();

  test_SSHconnection( "echo stdout read [stdout]", false );
  test_SSHconnection( "echo stderr read [stderr] 1>&2", false );

  test_SSHconnection( "echo capture stdout [stdout]", true );
  test_SSHconnection( "echo capture stdout [stdout]; echo but not capture stderr [stderr] 1>&2", true );

  test_SSHconnection( "echo stderr first [stderr] 1>&2; echo stdout second [stdout]", false );
  test_SSHconnection( "echo stdout first [stdout]; echo stderr second [stderr] 1>&2", false );

  SSH_Finalize();

  return 0;
}

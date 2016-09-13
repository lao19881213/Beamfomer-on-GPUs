//# SSH.cc: set up an SSH connection using libssh2
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: SSH.cc 25781 2013-07-26 11:48:54Z mol $

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include "SSH.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>
#include <sys/select.h>

#ifdef HAVE_OPENSSL
#include <openssl/crypto.h>
#endif

#ifdef HAVE_GCRYPT
#include <pthread.h>
#include <gcrypt.h>
GCRY_THREAD_OPTION_PTHREAD_IMPL;
#endif

#include <Common/LofarLogger.h>
#include <Common/SystemCallException.h>
#include <Common/Thread/Cancellation.h>
#include <Common/Thread/Mutex.h>
#include <Stream/SocketStream.h>

using namespace std;

namespace LOFAR
{
  namespace Cobalt
  {

#ifndef HAVE_LOG4COUT
    Mutex coutMutex;
#endif

    // Prevent export of free_session and free_channel. They
    // cannot be made static as they are needed as a template
    // parameter.
    namespace
    {

      // free a LIBSSH2_SESSION object
      void free_session( LIBSSH2_SESSION *session )
      {
        ScopedDelayCancellation dc;

        if (!session)
          return;

        libssh2_session_free(session);
      }

      // free a LIBSSH2_CHANNEL object
      void free_channel( LIBSSH2_CHANNEL *channel )
      {
        ScopedDelayCancellation dc;

        if (!channel)
          return;

        libssh2_channel_free(channel);
      }

      typedef SmartPtr<LIBSSH2_SESSION, SmartPtrFreeFunc<LIBSSH2_SESSION, free_session> > session_t;
      typedef SmartPtr<LIBSSH2_CHANNEL, SmartPtrFreeFunc<LIBSSH2_CHANNEL, free_channel> > channel_t;

    }


    // Convert an SSH exit status to a string
    static const char *explainExitStatus( int exitstatus )
    {
      const char *explanation;

      switch (exitstatus) {
      default:
        explanation = "??";
        break;

      case 255:
        explanation = "Network or authentication error";
        break;
      case 127:
        explanation = "BASH: command/library not found";
        break;
      case 126:
        explanation = "BASH: command found but could not be executed (wrong architecture?)";
        break;

      case 128 + SIGHUP:
        explanation = "killed by SIGHUP";
        break;
      case 128 + SIGINT:
        explanation = "killed by SIGINT (Ctrl-C)";
        break;
      case 128 + SIGQUIT:
        explanation = "killed by SIGQUIT";
        break;
      case 128 + SIGILL:
        explanation = "illegal instruction";
        break;
      case 128 + SIGABRT:
        explanation = "killed by SIGABRT";
        break;
      case 128 + SIGKILL:
        explanation = "killed by SIGKILL";
        break;
      case 128 + SIGSEGV:
        explanation = "segmentation fault";
        break;
      case 128 + SIGPIPE:
        explanation = "broken pipe";
        break;
      case 128 + SIGALRM:
        explanation = "killed by SIGALRM";
        break;
      case 128 + SIGTERM:
        explanation = "killed by SIGTERM";
        break;
      }

      return explanation;
    }


    // Convert a LibSSH2 error code to a string
    static string explainLibSSH2Error( LIBSSH2_SESSION *session, int error )
    {
      string error_str;

      // convert error code
      switch(error) {
      default:
        error_str = "??";
        break;

      case LIBSSH2_ERROR_NONE:                          error_str = "LIBSSH2_ERROR_NONE";
        break;
      case LIBSSH2_ERROR_SOCKET_NONE:                   error_str = "LIBSSH2_ERROR_SOCKET_NONE";
        break;
#if LIBSSH2_VERSION_NUM > 0x010207
      case LIBSSH2_ERROR_BANNER_RECV:                   error_str = "LIBSSH2_ERROR_BANNER_RECV";
        break;
#else
      case LIBSSH2_ERROR_BANNER_NONE:                   error_str = "LIBSSH2_ERROR_BANNER_NONE";
        break;
#endif
      case LIBSSH2_ERROR_BANNER_SEND:                   error_str = "LIBSSH2_ERROR_BANNER_SEND";
        break;
      case LIBSSH2_ERROR_INVALID_MAC:                   error_str = "LIBSSH2_ERROR_INVALID_MAC";
        break;
      case LIBSSH2_ERROR_KEX_FAILURE:                   error_str = "LIBSSH2_ERROR_KEX_FAILURE";
        break;
      case LIBSSH2_ERROR_ALLOC:                         error_str = "LIBSSH2_ERROR_ALLOC";
        break;
      case LIBSSH2_ERROR_SOCKET_SEND:                   error_str = "LIBSSH2_ERROR_SOCKET_SEND";
        break;
      case LIBSSH2_ERROR_KEY_EXCHANGE_FAILURE:          error_str = "LIBSSH2_ERROR_KEY_EXCHANGE_FAILURE";
        break;
      case LIBSSH2_ERROR_TIMEOUT:                       error_str = "LIBSSH2_ERROR_TIMEOUT";
        break;
      case LIBSSH2_ERROR_HOSTKEY_INIT:                  error_str = "LIBSSH2_ERROR_HOSTKEY_INIT";
        break;
      case LIBSSH2_ERROR_HOSTKEY_SIGN:                  error_str = "LIBSSH2_ERROR_HOSTKEY_SIGN";
        break;
      case LIBSSH2_ERROR_DECRYPT:                       error_str = "LIBSSH2_ERROR_DECRYPT";
        break;
      case LIBSSH2_ERROR_SOCKET_DISCONNECT:             error_str = "LIBSSH2_ERROR_SOCKET_DISCONNECT";
        break;
      case LIBSSH2_ERROR_PROTO:                         error_str = "LIBSSH2_ERROR_PROTO";
        break;
      case LIBSSH2_ERROR_PASSWORD_EXPIRED:              error_str = "LIBSSH2_ERROR_PASSWORD_EXPIRED";
        break;
      case LIBSSH2_ERROR_FILE:                          error_str = "LIBSSH2_ERROR_FILE";
        break;
      case LIBSSH2_ERROR_METHOD_NONE:                   error_str = "LIBSSH2_ERROR_METHOD_NONE";
        break;
      case LIBSSH2_ERROR_AUTHENTICATION_FAILED:         error_str = "LIBSSH2_ERROR_AUTHENTICATION_FAILED";
        break;
      //case LIBSSH2_ERROR_PUBLICKEY_UNRECOGNIZED:	error_str = "LIBSSH2_ERROR_PUBLICKEY_UNRECOGNIZED"; break;
      case LIBSSH2_ERROR_PUBLICKEY_UNVERIFIED:          error_str = "LIBSSH2_ERROR_PUBLICKEY_UNVERIFIED";
        break;
      case LIBSSH2_ERROR_CHANNEL_OUTOFORDER:            error_str = "LIBSSH2_ERROR_CHANNEL_OUTOFORDER";
        break;
      case LIBSSH2_ERROR_CHANNEL_FAILURE:               error_str = "LIBSSH2_ERROR_CHANNEL_FAILURE";
        break;
      case LIBSSH2_ERROR_CHANNEL_REQUEST_DENIED:        error_str = "LIBSSH2_ERROR_CHANNEL_REQUEST_DENIED";
        break;
      case LIBSSH2_ERROR_CHANNEL_UNKNOWN:               error_str = "LIBSSH2_ERROR_CHANNEL_UNKNOWN";
        break;
      case LIBSSH2_ERROR_CHANNEL_WINDOW_EXCEEDED:       error_str = "LIBSSH2_ERROR_CHANNEL_WINDOW_EXCEEDED";
        break;
      case LIBSSH2_ERROR_CHANNEL_PACKET_EXCEEDED:       error_str = "LIBSSH2_ERROR_CHANNEL_PACKET_EXCEEDED";
        break;
      case LIBSSH2_ERROR_CHANNEL_CLOSED:                error_str = "LIBSSH2_ERROR_CHANNEL_CLOSED";
        break;
      case LIBSSH2_ERROR_CHANNEL_EOF_SENT:              error_str = "LIBSSH2_ERROR_CHANNEL_EOF_SENT";
        break;
      case LIBSSH2_ERROR_SCP_PROTOCOL:                  error_str = "LIBSSH2_ERROR_SCP_PROTOCOL";
        break;
      case LIBSSH2_ERROR_ZLIB:                          error_str = "LIBSSH2_ERROR_ZLIB";
        break;
      case LIBSSH2_ERROR_SOCKET_TIMEOUT:                error_str = "LIBSSH2_ERROR_SOCKET_TIMEOUT";
        break;
      case LIBSSH2_ERROR_SFTP_PROTOCOL:                 error_str = "LIBSSH2_ERROR_SFTP_PROTOCOL";
        break;
      case LIBSSH2_ERROR_REQUEST_DENIED:                error_str = "LIBSSH2_ERROR_REQUEST_DENIED";
        break;
      case LIBSSH2_ERROR_METHOD_NOT_SUPPORTED:          error_str = "LIBSSH2_ERROR_METHOD_NOT_SUPPORTED";
        break;
      case LIBSSH2_ERROR_INVAL:                         error_str = "LIBSSH2_ERROR_INVAL";
        break;
      case LIBSSH2_ERROR_INVALID_POLL_TYPE:             error_str = "LIBSSH2_ERROR_INVALID_POLL_TYPE";
        break;
      case LIBSSH2_ERROR_PUBLICKEY_PROTOCOL:            error_str = "LIBSSH2_ERROR_PUBLICKEY_PROTOCOL";
        break;
      case LIBSSH2_ERROR_EAGAIN:                        error_str = "LIBSSH2_ERROR_EAGAIN";
        break;
      case LIBSSH2_ERROR_BUFFER_TOO_SMALL:              error_str = "LIBSSH2_ERROR_BUFFER_TOO_SMALL";
        break;
      case LIBSSH2_ERROR_BAD_USE:                       error_str = "LIBSSH2_ERROR_BAD_USE";
        break;
      case LIBSSH2_ERROR_COMPRESS:                      error_str = "LIBSSH2_ERROR_COMPRESS";
        break;
      case LIBSSH2_ERROR_OUT_OF_BOUNDARY:               error_str = "LIBSSH2_ERROR_OUT_OF_BOUNDARY";
        break;
      case LIBSSH2_ERROR_AGENT_PROTOCOL:                error_str = "LIBSSH2_ERROR_AGENT_PROTOCOL";
        break;
      case LIBSSH2_ERROR_SOCKET_RECV:                   error_str = "LIBSSH2_ERROR_SOCKET_RECV";
        break;
      case LIBSSH2_ERROR_ENCRYPT:                       error_str = "LIBSSH2_ERROR_ENCRYPT";
        break;
      case LIBSSH2_ERROR_BAD_SOCKET:                    error_str = "LIBSSH2_ERROR_BAD_SOCKET";
        break;
        //      case LIBSSH2_ERROR_KNOWN_HOSTS:			error_str = "LIBSSH2_ERROR_KNOWN_HOSTS"; break;
      }

      // ask libssh2 for more info
      if (session) {
        char *errormsg = NULL;

        libssh2_session_last_error(session, &errormsg, NULL, 0);

        return error_str + ": " + errormsg;
      }

      return error_str;
    }

    /*
     * Make sure we ScopedDelayCancellation around calls to libssh2 because
     * it is an external library.
     *
     * Note that waitsocket() is a forced cancellation point.
     */

    SSHconnection::SSHconnection(const string &logPrefix, const string &hostname, const string &commandline, const string &username, const string &pubkey, const string &privkey, bool captureStdout, ostream &_cout, ostream &_cerr)
      :
      itsLogPrefix(logPrefix),
      itsHostName(hostname),
      itsCommandLine(commandline),
      itsUserName(username),
      itsPublicKey(pubkey),
      itsPrivateKey(privkey),
      itsConnected(false),
      itsCaptureStdout(captureStdout),
      itsCout(_cout),
      itsCerr(_cerr)
    {
    }


    SSHconnection::~SSHconnection()
    {
      if (itsThread.get())
        cancel();
    }


    void SSHconnection::start()
    {
      itsThread = new Thread(this, &SSHconnection::commThread, itsLogPrefix + "[SSH Thread] ", 65536);
    }


    bool SSHconnection::isDone()
    {
      return itsThread && itsThread->isDone();
    }


    bool SSHconnection::connected() const
    {
      return itsConnected;
    }


    void SSHconnection::cancel()
    {
      ASSERT(itsThread.get());

      itsThread->cancel();

      itsThread->wait();
    }


    void SSHconnection::wait()
    {
      ASSERT(itsThread.get());

      itsThread->wait();
    }


    void SSHconnection::wait( const struct timespec &deadline )
    {
      ASSERT(itsThread.get());

      if (!itsThread->wait(deadline)) {
        itsThread->cancel();

        itsThread->wait();
      }
    }


    std::string SSHconnection::stdoutBuffer() const
    {
      return itsStdoutBuffer.str();
    }

    LIBSSH2_SESSION *SSHconnection::open_session( FileDescriptorBasedStream &sock )
    {
      ScopedDelayCancellation dc;

      int rc;

      /* Create a session instance */
      session_t session = libssh2_session_init();
      if (!session.get()) {
        LOG_ERROR_STR( itsLogPrefix << "Cannot create SSH session object" );
        return 0;
      }

      /* tell libssh2 we want it all done non-blocking */
      libssh2_session_set_blocking(session, 0);

      /* ... start it up. This will trade welcome banners, exchange keys,
       * and setup crypto, compression, and MAC layers
       */
#if LIBSSH2_VERSION_NUM >= 0x010208
      while ((rc = libssh2_session_handshake(session, sock.fd)) ==
             LIBSSH2_ERROR_EAGAIN) {
        waitsocket(session, sock);
      }
#else
      while ((rc = libssh2_session_startup(session, sock.fd)) ==
             LIBSSH2_ERROR_EAGAIN) {
        waitsocket(session, sock);
      }
#endif

      /* NOTE: libssh2 now holds a copy of sock.fd, so don't invalidate it! */

      if (rc) {
        LOG_ERROR_STR( itsLogPrefix << "Failure establishing SSH session: " << rc << " (" << explainLibSSH2Error(session, rc) << ")");
        return NULL;
      }

      /* Authenticate by public and/or private key */
      while ((rc = libssh2_userauth_publickey_fromfile(session,
                                                       itsUserName.c_str(), // remote username
                                                       itsPublicKey == "" ? NULL : itsPublicKey.c_str(), // public key filename
                                                       itsPrivateKey == "" ? NULL : itsPrivateKey.c_str(), // private key filename
                                                       NULL // password
                                                       )) ==
             LIBSSH2_ERROR_EAGAIN) {
        waitsocket(session, sock);
      }

      if (rc) {
        LOG_ERROR_STR( itsLogPrefix << "Authentication for user '" << itsUserName << "' by public/private keys '" << itsPublicKey << "'/'" << itsPrivateKey << "' failed: " << rc << " (" << explainLibSSH2Error(session, rc) << ")");
        return NULL;
      }

      return session.release();
    }

    void SSHconnection::close_session( LIBSSH2_SESSION *session, FileDescriptorBasedStream &sock )
    {
      ScopedDelayCancellation dc;

      int rc;

      while ((rc = libssh2_session_disconnect(session, "Normal Shutdown, Thank you for playing")) ==
             LIBSSH2_ERROR_EAGAIN) {
        waitsocket(session, sock);
      }

      if (rc)
      {
        LOG_ERROR_STR( itsLogPrefix << "Failure closing session: " << rc << " (" << explainLibSSH2Error(session, rc) << ")");
        return;
      }
    }

    LIBSSH2_CHANNEL *SSHconnection::open_channel( LIBSSH2_SESSION *session, FileDescriptorBasedStream &sock )
    {
      ScopedDelayCancellation dc;

      channel_t channel;

      /* Exec non-blocking on the remote host */
      while( (channel = libssh2_channel_open_session(session)) == NULL &&
             libssh2_session_last_error(session,NULL,NULL,0) ==
             LIBSSH2_ERROR_EAGAIN )
      {
        waitsocket(session, sock);
      }

      if (!channel.get())
      {
        LOG_ERROR_STR( itsLogPrefix << "Could not set up SSH channel" );
        return NULL;
      }

      return channel.release();
    }

    void SSHconnection::close_channel( LIBSSH2_SESSION *session, LIBSSH2_CHANNEL *channel, FileDescriptorBasedStream &sock )
    {
      ScopedDelayCancellation dc;

      int rc;

      while( (rc = libssh2_channel_close(channel)) == LIBSSH2_ERROR_EAGAIN ) {
        waitsocket(session, sock);
      }

      if (rc)
      {
        LOG_ERROR_STR( itsLogPrefix << "Failure closing channel: " << rc << " (" << explainLibSSH2Error(session, rc) << ")");
        return;
      }
    }

    bool SSHconnection::waitsocket( LIBSSH2_SESSION *session, FileDescriptorBasedStream &sock )
    {
      // we manually control the cancellation points, so make sure
      // cancellation is actually disabled.
      ScopedDelayCancellation dc;

      struct timeval timeout;
      int rc;
      fd_set fd;
      fd_set *writefd = NULL;
      fd_set *readfd = NULL;
      int dir;

      timeout.tv_sec = 1;
      timeout.tv_usec = 0;

      FD_ZERO(&fd);

      FD_SET(sock.fd, &fd);

      /* now make sure we wait in the correct direction */
      dir = libssh2_session_block_directions(session);

      if(dir & LIBSSH2_SESSION_BLOCK_INBOUND)
        readfd = &fd;

      if(dir & LIBSSH2_SESSION_BLOCK_OUTBOUND)
        writefd = &fd;

      {
        Cancellation::enable();

        // select() is a cancellation point
        rc = ::select(sock.fd + 1, readfd, writefd, NULL, &timeout);

        Cancellation::disable();
      }

      return rc > 0;
    }

    void SSHconnection::commThread()
    {
      int rc;
      int exitcode;
      char *exitsignal = 0;

      // WARNING: Make sure sock stays alive while a session is active, because the session
      // will retain a copy of sock.fd so we can't invalidate it. We don't want to
      // (for example) send a libssh2_session_disconnect to a sock.fd that has been
      // reused by the system!

      // Declaring sock before session will cause ~sock to be called after
      // ~session.

      SmartPtr<SocketStream> sock;
      session_t session;
      channel_t channel;

      for(;; ) {
        // keep trying to connect
        sock = new SocketStream( itsHostName, 22, SocketStream::TCP, SocketStream::Client, 0 );

        LOG_DEBUG_STR( itsLogPrefix << "Connected" );

        /* Prevent cancellation in functions dealing with libssh2 internals, but
         * NOT during sleep() */
        {
          {
            ScopedDelayCancellation dc;

            session = open_session(*sock);
          }

          if (session.get()) {
            ScopedDelayCancellation dc;

            channel = open_channel(session, *sock);

            if (channel.get())
              // success!
              break;

            close_session(session, *sock);

            session = 0;
          }

          sleep(RETRY_DELAY);
        }

        break;
      }

      itsConnected = true;

      LOG_DEBUG_STR( itsLogPrefix << "Starting remote command: " << itsCommandLine);

      while( (rc = libssh2_channel_exec(channel, itsCommandLine.c_str())) ==
             LIBSSH2_ERROR_EAGAIN )
      {
        waitsocket(session, *sock);
      }

      if (rc)
      {
        LOG_ERROR_STR( itsLogPrefix << "Failure starting remote command: " << rc << " (" << explainLibSSH2Error(session, rc) << ")");
        return;
      }

      LOG_DEBUG_STR( itsLogPrefix << "Remote command started, waiting for output" );

      Cancellation::disable();
      Cancellation::point();

#define NRSTREAMS 2

      // raw input buffer
      char data[NRSTREAMS][0x1000];

      // the current line (or line remnant)
      string line[NRSTREAMS];

      // how many streams still provide data
      unsigned nrOpenStreams = NRSTREAMS;

      // which streams still provide data
      bool isOpen[NRSTREAMS];

      for (unsigned s = 0; s < NRSTREAMS; ++s)
        isOpen[s] = true;

      /* Session I/O */
      while( nrOpenStreams > 0 )
      {
        for (unsigned s = 0; s < NRSTREAMS; ++s) {
          if (!isOpen[s])
            continue;

          /* loop until we block */
          do {
            rc = libssh2_channel_read_ex(channel, s, data[s], sizeof data[s]);
            if( rc > 0 )
            {
              if (s == 0 && itsCaptureStdout) {
                // save stdout verbatim in our buffer

                LOG_DEBUG_STR( itsLogPrefix << "Appending " << rc << " bytes to stdout buffer, which contains " << itsStdoutBuffer.rdbuf()->in_avail() << " bytes" );

                itsStdoutBuffer.write( data[s], rc );
              } else {
                // print stream to stdout (TODO: to logger)

                // create a buffer for line + data
                stringstream buffer;

                buffer << line[s];
                buffer.write( data[s], rc );

                /* extract and log lines */
                for(;; )
                {
                  Cancellation::point();

                  std::getline( buffer, line[s] );

                  if (!buffer.good()) {
                    // 'line' now holds the remnant

                    if (line[s].size() > 10240) {
                      LOG_ERROR_STR( itsLogPrefix << "Line too long (" << line[s].size() << "); truncated: " << line[s] );
                      line[s] = "";
                    }
                    break;
                  }

                  // TODO: Use logger somehow (we'd duplicate the prefix if we just use LOG_* macros..)
                  {
#ifdef HAVE_LOG4COUT
                    ScopedLock sl(LFDebug::mutex);
#else
                    ScopedLock sl(coutMutex);
#endif
                    if (s == 0)
                      itsCout << line[s] << endl;
                    else
                      itsCerr << line[s] << endl;
                  }
                }
              }
            } else {
              if( rc < 0 && rc != LIBSSH2_ERROR_EAGAIN ) {
                /* no need to output this for the EAGAIN case */
                LOG_ERROR_STR( itsLogPrefix << "libssh2_channel_read_ex returned " << rc << " (" << explainLibSSH2Error(session, rc) << ") for channel " << s);
              }
            }
          } while( rc > 0 );

          /* this is due to blocking that would occur otherwise so we loop on
             this condition */
          if( rc != LIBSSH2_ERROR_EAGAIN )
          {
            /* EOF */
            --nrOpenStreams;
          }
        }

        if (nrOpenStreams > 0)
          waitsocket(session, *sock);
      }

      LOG_DEBUG_STR( itsLogPrefix << "Disconnecting" );

      close_channel(session, channel, *sock);

      if (rc == 0)
      {
        exitcode = libssh2_channel_get_exit_status(channel);
#if LIBSSH2_VERSION_NUM >= 0x010208
        libssh2_channel_get_exit_signal(channel, &exitsignal,
                                        NULL, NULL, NULL, NULL, NULL);
#else
        exitsignal = 0;
#endif
      } else {
        exitcode = 127;
      }

      close_session(session, *sock);

      if (exitsignal) {
        LOG_ERROR_STR(itsLogPrefix << "SSH was killed by signal " << exitsignal);
      } else if(exitcode > 0) {
        LOG_ERROR_STR(itsLogPrefix << "Exited with exit code " << exitcode << " (" << explainExitStatus(exitcode) << ")" );
      } else {
        LOG_INFO_STR(itsLogPrefix << "Terminated normally");
      }
    }



#ifdef HAVE_OPENSSL
    std::vector< SmartPtr<Mutex> > openssl_mutexes;

    static void lock_callback(int mode, int type, const char *file, int line)
    {
      (void)file;
      (void)line;

      if (mode & CRYPTO_LOCK)
        openssl_mutexes[type]->lock();
      else
        openssl_mutexes[type]->unlock();
    }

    static unsigned long thread_id_callback()
    {
      return static_cast<unsigned long>(pthread_self());
    }
#endif

    bool SSH_Init()
    {
#ifdef HAVE_OPENSSL
      // initialise openssl
      openssl_mutexes.resize(CRYPTO_num_locks());
      for (size_t i = 0; i < openssl_mutexes.size(); ++i)
        openssl_mutexes[i] = new Mutex;

      CRYPTO_set_id_callback(&thread_id_callback);
      CRYPTO_set_locking_callback(&lock_callback);
#endif

#ifdef HAVE_GCRYPT
      // initialise gcrypt
      gcry_control(GCRYCTL_SET_THREAD_CBS, &gcry_threads_pthread);
#endif

#if LIBSSH2_VERSION_NUM >= 0x010205
      // initialise libssh2
      int rc = libssh2_init(0);

      if (rc)
        return false;
#endif

      return true;
    }

    void SSH_Finalize()
    {
#if LIBSSH2_VERSION_NUM >= 0x010205
      // exit libssh2
      libssh2_exit();
#endif

#ifdef HAVE_OPENSSL
      // exit openssl
      CRYPTO_set_locking_callback(NULL);
      CRYPTO_set_id_callback(NULL);

      openssl_mutexes.clear();
#endif
    }


    // Returns true if the given private-key file name works for "ssh localhost"
    static bool ssh_works(const char *pubkey, const char *privkey)
    {
      char *USER = getenv("USER");

      ASSERTSTR(USER, "$USER not set");

      // connect, running /bin/true
      SSHconnection sshconn("", "localhost", "/bin/true", USER, pubkey, privkey, true);
      sshconn.start();

      // wait 5 seconds for connection to succeed
      struct timespec deadline = { time(0) + 5, 0 };
      sshconn.wait(deadline);

      // return whether connection succeeded
      return sshconn.connected();
    }


    bool discover_ssh_keys(char *pubkey, size_t pubkey_buflen, char *privkey, size_t privkey_buflen)
    {
      ASSERT(pubkey_buflen > 0);
      ASSERT(privkey_buflen > 0);

      char *HOME = getenv("HOME");

      ASSERTSTR(HOME, "$HOME not set");

      // try several common keys
      for(unsigned attempt = 0;; attempt++) {
        switch (attempt) {
        case 0:
          pubkey[0] = 0;
          snprintf(privkey, privkey_buflen, "%s/.ssh/id_dsa",     HOME);
          break;

        case 1:
          snprintf(pubkey,  pubkey_buflen,  "%s/.ssh/id_dsa.pub", HOME);
          snprintf(privkey, privkey_buflen, "%s/.ssh/id_dsa",     HOME);
          break;

        case 2:
          pubkey[0] = 0;
          snprintf(privkey, privkey_buflen, "%s/.ssh/id_rsa",     HOME);
          break;

        case 3:
          snprintf(pubkey,  pubkey_buflen,  "%s/.ssh/id_rsa.pub", HOME);
          snprintf(privkey, privkey_buflen, "%s/.ssh/id_rsa",     HOME);
          break;

        default:
          // ran out of attempts
          LOG_ERROR("Cannot find a working public/private key for SSH to localhost");
          return false;
        }

        // try key pair
        if (ssh_works(pubkey, privkey)) {
          LOG_DEBUG_STR("Key files '" << pubkey << "' and '" << privkey << "' work for ssh localhost.");
          return true;
        }
      }
    }

  } // namespace Cobalt
} // namespace LOFAR


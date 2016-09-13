#!/usr/bin/python

import sys
import os
from time import strftime,localtime,sleep
import logging
from logging.handlers import TimedRotatingFileHandler
from traceback import format_exception
from itertools import count
import socket
from struct import pack
import Queue
from threading import Thread

DEBUG=False

class reconnecting_socket:
  """ A socket that keeps reconnecting if the connection is lost. Data is sent
      asynchronously, with a buffer which drops messages if full. """

  def __init__( self, host, port, retry_timeout=10, socket_timeout=5, bufsize=256 ):
    self.host = host
    self.port = port
    self.socket_timeout = socket_timeout
    self.retry_timeout = retry_timeout
    self.socket = None
    self.done = False

    self.writebuf = Queue.Queue( bufsize )

    self.iothread = Thread( target=self.iothread_main, name="I/O thread for %s:%s" % (host,port) )
    self.iothread.start()

  def iothread_main( self ):
    def close():
      self.socket.close()
      self.socket = None

    def reconnect():
      self.socket = socket.socket()
      self.socket.settimeout( self.socket_timeout )

      while not self.done:
        try:
          self.socket.connect( (self.host,self.port) )
          self.socket.setsockopt( socket.SOL_SOCKET, socket.SO_LINGER, pack('ii', 1, self.socket_timeout) )
        except socket.error:
          pass
        except socket.timeout:
          pass
        else:
          # connected!
          break

        # sleep, but do stop when told
        for i in xrange( self.retry_timeout ):
          if self.done:
            return
          sleep( 1 )

    def write( data ):
      written = 0

      while written < len(data):
        if self.socket is None:
          reconnect()

        if self.done:
          return

        try:
          written += self.socket.send( data[written:] )
        except socket.error:
          close()
          continue
        except socket.timeout:
          close()
	  continue

    # start with a connection
    if self.socket is None:
      reconnect()

    while True:
      try:
        data = self.writebuf.get( timeout=1 )
      except Queue.Empty:
        # TODO: we can't keep a close check on our socket, delaying
        # closing and reconnecting and keeping the line open
        continue

      if data is None:
        # close request
        break

      write( data ) 

  def write( self, data ):
    if self.done:
      return

    try:
      self.writebuf.put_nowait( data )
    except Queue.Full:
      # queue full -- drop data
      pass

  def close( self ):
    self.done = True # abort any reconnection attempts
    self.writebuf.put( None ) # prod the deque, wait if necessary

    self.iothread.join()

def my_excepthook( etype, value, tb ):
  """ Replacement for default exception handler, which uses the logger instead of stderr. """

  lines = format_exception( etype, value, tb )

  for l in lines:
    for m in l.split("\n")[:-1]:
      logging.critical( m )

def initLogger():
  if DEBUG:
    minloglevel = logging.DEBUG
  else:
    minloglevel = logging.INFO

  logging.basicConfig( level = minloglevel,
                       format = "OLAP %(asctime)s.%(msecs)03d %(levelname)-5s %(message)s",
                       datefmt = "%d-%m-%y %H:%M:%S",
                   )

  logging.raiseExceptions = False                 

  loglevels = {
   "DEBUG":  logging.DEBUG,
   "INFO":   logging.INFO,
   "WARN":   logging.WARNING,
   "ERROR":  logging.ERROR,
   "FATAL":  logging.CRITICAL
  }

  for name,level in loglevels.iteritems():
    logging.addLevelName( level, name )

  sys.excepthook = my_excepthook  

class TimedSizeRotatingFileHandler(TimedRotatingFileHandler):
  """
    Rolls over both at midnight and after the log has reached a certain size.
  """
  def __init__(self, filename):
    TimedRotatingFileHandler.__init__( self, filename, when = "midnight", interval = 1, backupCount = 0 )

    self.maxBytes = 1024*1024*1024
    self.fileCount = 1

  def shouldRollover(self, record):
    if TimedRotatingFileHandler.shouldRollover(self, record):
      return 1

    msg = "%s\n" % self.format(record)

    if self.stream.tell() + len(msg) >= self.maxBytes:
      return 1

    return 0  

  def doRollover(self):
    def rename(f,t):
      if not os.path.exists(f):
        return

      if os.path.exists(t):
        os.remove(t)
      os.rename(f,t)

    t = self.rolloverAt - self.interval
    timeTuple = localtime(t)

    base = self.baseFilename + "." + strftime(self.suffix, timeTuple)

    if os.path.exists( base ):
      # increment the suffix number of older logfiles, since TimedRotatingFileHandler
      # will overwrite "base"
      for b in xrange( self.fileCount, -1, -1 ):
        if b == 0:
          sfn = base
        else:  
          sfn = base + "." + str(b)

        dfn = base + "." + str(b+1)

        rename( sfn, dfn )

      self.fileCount += 1  
    else:
      self.fileCount = 1

    TimedRotatingFileHandler.doRollover(self)

def rotatingLogger( appname, filename ):
  logger = logging.getLogger( appname )

  handler = TimedSizeRotatingFileHandler( filename )

  logger.propagate = False
  logger.addHandler( handler )

  return logger

if __name__ == "__main__":
  import sys

  if len(sys.argv) < 2:
    print "Usage: %s outputfilename [maxfilesize]" % (sys.argv[0],)
    sys.exit(1)

if __name__ == "__main__":
  from optparse import OptionParser,OptionGroup

  parser = OptionParser( usage = """usage: %prog [options] outputfilename
    """ )

  parser.add_option( "-s", "--server",
  			dest = "server",
			type = "string",
  			help = "output to logserver (host:port)" )
  parser.add_option( "-v", "--verbose",
  			dest = "verbose",
			action = "store_true",
			default = False,
  			help = "output to stdout [%default]" )
  parser.add_option( "-t", "--timestamp",
  			dest = "timestamp",
			action = "store_true",
			default = False,
  			help = "prefix each line with the current date/time [%default]" )
  parser.add_option( "-m", "--maxmb",
  			dest = "maxmb",
			type = "int",
                        default = 512,
  			help = "maximum file size in megabytes [%default]" )

  # parse arguments
  (options, args) = parser.parse_args()

  if not args:
    parser.print_help()
    sys.exit(1)

  initLogger()

  logfilename = args[0]
  logger = rotatingLogger( "foo", logfilename )
  logger.handlers[0].maxBytes = options.maxmb * 1024 * 1024

  verbose = options.verbose
  add_time = options.timestamp
  if options.server:
    host,port = options.server.split(":")
    port = int(port)

    if port == 0:
      print "Invalid port number: %s" % (sys.argv[2],)
      sys.exit(1)

    server = reconnecting_socket(host, port)
  else:
    server = None

  # 'for line in sys.stdin' buffers input, which
  # is not what we want at all, so we use
  # sys.stdin.readline instead.
  for line in iter(sys.stdin.readline, ""): 
    if server:
      server.write(line)

    line = line[:-1] # strip trailing \n

    if add_time:
      line = strftime("%Y-%m-%d %H:%M:%S ") + line

    logger.info( "%s", line )

    if verbose:
      print line


  if server:
    server.close()


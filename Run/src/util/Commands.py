from subprocess import Popen,STDOUT,PIPE
from Aborter import runFunc

# define our own PIPE as an alias to subprocess.PIPE
PIPE=PIPE

def backquote( cmdline, timeout = 0 ):
  """ Run a command line and return the output. """

  debug("RUN backquote: %s" % (cmdline,) )
  class process:
    def __init__(self):
      self.output = ""

    def run(self):
      self.output = Popen( cmdline.split(), stdout=PIPE, stderr=STDOUT ).communicate()[0]

  p = process()
  if timeout:
    runFunc( p.run, timeout )
  else:
    p.run()

  return p.output

def debug( str ):
  """ Override with custom logging function. """

  pass

#!/usr/bin/env python

from util.Hosts import ropen

# do not modify any files if DRYRUN is True
DRYRUN = False

"""
  The following files exist to aid the creation of observation IDs:

  nextMSnumber	contains the next free observation ID (integer)
"""

class ObservationID:
  def __init__( self ):
    self.obsid = 0

  def generateID( self, nextMSnumber = "/globalhome/lofarsystem/log/nextMSNumber" ):
    """ Returns an unique observation ID to use and reserve it. """

    if self.obsid:
      # already reserved an observation ID
      return self.obsid

    # read the next ms number
    f  = ropen( nextMSnumber, "r" )
    obsid = int(f.readline())
    f.close()

    if not DRYRUN:
      # increment it and save
      f = ropen( nextMSnumber, "w" )
      print >>f, "%s" % (obsid+1)
      f.close()

    self.obsid = obsid

    return self.obsid

if __name__ == "__main__":
  obsID = ObservationID()
  print obsID.generateID()


#!/usr/bin/env python

# Script for debugging addbeaminfo
# deletes entries and sets element flags back
#
# File:         debugbeaminfo.py
# Author:       Sven Duscha (duscha@astron.nl)
# Date:         2011-12-06
# Last change:  2011-12-06


import sys
import pyrap.tables as pt

MS="/Users/duscha/Cluster/L2011_24380/L24380_SB030_uv.MS.dppp.dppp"


# Remove LOFAR_FAILED_ELEMENTS entries
#
def removeFailedElements(antennaFieldId):
  print "removeFailedElements()"  # DEBUG

  failedElementsTab=pt.table(MS+"/LOFAR_ELEMENT_FAILURE", readonly=False)
  nrows=failedElementsTab.nrows()

  print MS+"/LOFAR_ELEMENT_FAILURE has nrows = ", nrows    # DEBUG
  
  if nrows > 0:
    print "removing rows 0 to ", nrows
    if antennaFieldId=="":     # remove all
      while nrows > 0:
        print "removing row =  ",  nrows
        failedElementsTab.removerows(nrows-1)
	nrows=failedElementsTab.nrows()
    else:                      # remove only those for particular station
      antennaFieldIdCol=failedElementsTab.getcol("ANTENNA_FIELD_ID")
      for i in range(0, nrows):
        if antennaFieldIdCol[i]==antennaFieldId:
          failedElementsTab.removerows(i)
  

# Set ELEMENT_FLAGS for particular antennaField from indexLow to
# indexHigh in the ELEMENT_FLAGS array
#
def setElementFlags(antennaFieldId, indexLow, indexHigh):
  print "setElementFlags()"     # DEBUG

  antennaFieldTab=pt.table(MS+"/LOFAR_ANTENNA_FIELD", readonly=False)
  
  # find ELEMENT_FLAGS array in row with corresponding antennaFieldId


def main():
  antennaFieldId=""

  # If we a command argument
  if len(sys.argv) > 1:
    antennaFieldId=sys.argv[1]
    
  tab=pt.table(MS, readonly=False)
  
  removeFailedElements(antennaFieldId)



if __name__=="__main__":
  main()

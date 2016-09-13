#!/usr/bin/env python

# Placeholder Python script for a DAL TBB validation tool.

import sys
import DAL

def main():
  if len(sys.argv) != 2:
    print 'Usage: ', sys.argv[0], 'L12345_xxx_tbb.h5'
    return 1

  filename = sys.argv[0]

  f = DAL.TBB_File(filename)


  print 'TBB file exists'
  return 0

if __name__ == '__main__':
  sys.exit(main())


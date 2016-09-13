//  MSMriterDAL.h: implementation of MSWriter using the DAL to write HDF5
//
//  Copyright (C) 2011
//  ASTRON (Netherlands Foundation for Research in Astronomy)
//  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//  $Id: MSWriterDAL.h 11891 2011-10-14 13:43:51Z gels $
//
//////////////////////////////////////////////////////////////////////


#ifndef LOFAR_STORAGE_MSWRITERDAL_H
#define LOFAR_STORAGE_MSWRITERDAL_H

//# Includes
#include <Common/LofarTypes.h>
#include <Common/ParameterSet.h>

#include <Interface/Parset.h>
#include <Interface/StreamableData.h>
#include <Storage/MSWriter.h>
#include <Storage/MSWriterFile.h>

#include <vector>

namespace LOFAR
{

  namespace RTCP
  {
    template<typename T, unsigned DIM> class MSWriterDAL : public MSWriterFile
    {
    public:
      MSWriterDAL(const string &filename, const Parset &parset, unsigned fileno, bool isBigEndian);
      ~MSWriterDAL();
      virtual void write(StreamableData *data);
    private:
      const Parset &itsParset;
      const Transpose2 &itsTransposeLogic;
      const StreamInfo &itsInfo;
      const unsigned itsNrChannels;
      const unsigned itsNrSamples;
      unsigned itsNextSeqNr;

      const unsigned itsBlockSize; // the size of StreamableData::samples, in T
    };
  }
}

#endif

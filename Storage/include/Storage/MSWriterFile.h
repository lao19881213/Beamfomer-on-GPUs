//  MSMriterNull.h: null implementation of MSWriter
//
//  Copyright (C) 2001
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
//  $Id: MSWriterImpl.h 11891 2008-10-14 13:43:51Z gels $
//
//////////////////////////////////////////////////////////////////////


#ifndef LOFAR_STORAGE_MSWRITERFILE_H
#define LOFAR_STORAGE_MSWRITERFILE_H


#include <Storage/MSWriter.h>
#include <Storage/FastFileStream.h>


namespace LOFAR {
namespace RTCP {


class MSWriterFile : public MSWriter
{
  public:
    MSWriterFile(const string &msName);
    ~MSWriterFile();

    virtual void write(StreamableData *data);

    virtual size_t getDataSize();

  protected:
    FastFileStream	 itsFile;
};


}
}

#endif

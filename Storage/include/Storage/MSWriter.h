//  MSMriter.h: Base class MSWriter
//
//  Copyright (C) 2008
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
//  $Id: MSWriter.h 22857 2012-11-20 08:58:52Z mol $
//
//////////////////////////////////////////////////////////////////////

#ifndef LOFAR_STORAGE_MSWRITER_H
#define LOFAR_STORAGE_MSWRITER_H

#include <Interface/StreamableData.h>
#include <Interface/FinalMetaData.h>
#include <Common/ParameterSet.h>

namespace LOFAR {
namespace RTCP {


class MSWriter
{
  public:
    MSWriter();
    virtual	 ~MSWriter();

    virtual void write(StreamableData *) = 0;

    virtual void augment(const FinalMetaData &finalMetaData);

    virtual size_t getDataSize();

    ParameterSet configuration() const;

    unsigned percentageWritten() const;

  protected:
    size_t itsNrBlocksWritten;
    size_t itsNrExpectedBlocks;
    ParameterSet itsConfiguration;
};


} // namespace RTCP
} // namespace LOFAR

#endif

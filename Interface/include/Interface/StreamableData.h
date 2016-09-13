#ifndef LOFAR_INTERFACE_STREAMABLE_DATA_H
#define LOFAR_INTERFACE_STREAMABLE_DATA_H

#include <Stream/Stream.h>
#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>
#include <Interface/Parset.h>
#include <Interface/MultiDimArray.h>
#include <Interface/SparseSet.h>
#include <Interface/Allocator.h>
#include <Interface/Align.h>
#include <Common/DataConvert.h>

#include <cstring>

namespace LOFAR {
namespace RTCP {

// Data which needs to be transported between CN, ION and Storage.
// Apart from read() and write() functionality, the data is augmented
// with a sequence number in order to detect missing data. Furthermore,
// an integration operator += can be defined to reduce the data.

// Endianness:
// * Endianness is defined by the correlator. 
// * Both Data and sequence number will have endianness of the
//   correlator
//
// WARNING: We consider all data streams to have the endianness of the
// correlator. No conversion is done here.

class IntegratableData
{
  public:
    virtual ~IntegratableData() {}

    virtual IntegratableData &operator += (const IntegratableData &) = 0;
};

    
class StreamableData
{
  public:
    static const uint32_t magic  = 0xda7a;
#ifdef HAVE_BGP
    static const size_t alignment = 32;
#else
    static const size_t alignment = 512;
#endif

    // the CPU which fills the datastructure sets the peerMagicNumber,
    // because other CPUs will overwrite it with a read(s,true) call from
    // either disk or network.
    StreamableData(): peerMagicNumber(magic), rawSequenceNumber(0) {}
    virtual ~StreamableData() {}

    void read(Stream *, bool withSequenceNumber, unsigned align = 0);
    void write(Stream *, bool withSequenceNumber, unsigned align = 0);

    bool shouldByteSwap() const
    { return peerMagicNumber != magic; }

    uint32_t sequenceNumber(bool raw=false) const {
      if (shouldByteSwap() && !raw) {
        uint32_t seqno = rawSequenceNumber;

        byteSwap32(&seqno);

        return seqno;
      } else {
        return rawSequenceNumber;
      }
    }

    void setSequenceNumber(uint32_t seqno) {
      if (shouldByteSwap())
        byteSwap32(&seqno);

      rawSequenceNumber = seqno;
    }

    virtual void setDimensions(unsigned, unsigned, unsigned) { }

    uint32_t peerMagicNumber;    /// magic number received from peer

  protected:
    // a subclass should override these to marshall its data
    virtual void readData(Stream *) = 0;
    virtual void writeData(Stream *) = 0;

  private:  
    uint32_t rawSequenceNumber;  /// possibly needs byte swapping
};


// A typical data set contains a MultiDimArray of tuples and a set of flags.
template <typename T = fcomplex, unsigned DIM = 4, unsigned FLAGS_DIM = 2> class SampleData : public StreamableData
{
  public:
    typedef typename MultiDimArray<T,DIM>::ExtentList ExtentList;
    typedef typename MultiDimArray<SparseSet<unsigned>,FLAGS_DIM>::ExtentList FlagsExtentList;

    SampleData(const ExtentList &extents, const FlagsExtentList &flagsExtents, Allocator & = heapAllocator);

    MultiDimArray<T,DIM>	      samples;
    MultiDimArray<SparseSet<unsigned>,FLAGS_DIM>   flags;

  protected:
    virtual void readData(Stream *);
    virtual void writeData(Stream *);

  private:
    //bool	 itsHaveWarnedLittleEndian;
};


inline void StreamableData::read(Stream *str, bool withSequenceNumber, unsigned alignment)
{
  if (withSequenceNumber) {
    std::vector<char> header(alignment > 2*sizeof(uint32_t) ? alignment : 2*sizeof(uint32_t));
    uint32_t          &magicValue = * reinterpret_cast<uint32_t *>(&header[0]);
    uint32_t	      &seqNo      = * reinterpret_cast<uint32_t *>(&header[sizeof(uint32_t)]);

    str->read(&header[0], header.size());

    peerMagicNumber = magicValue;
    rawSequenceNumber = seqNo;
  }

  readData(str);
}


inline void StreamableData::write(Stream *str, bool withSequenceNumber, unsigned alignment)
{
  
  if (withSequenceNumber) {
/*     std::vector<char> header(alignment > sizeof(uint32_t) ? alignment : sizeof(uint32_t)); */
    std::vector<char> header(alignment > 2*sizeof(uint32_t) ? alignment : 2*sizeof(uint32_t));
    uint32_t          &magicValue = * reinterpret_cast<uint32_t *>(&header[0]);
    uint32_t	      &seqNo      = * reinterpret_cast<uint32_t *>(&header[sizeof(uint32_t)]);

#if defined USE_VALGRIND
    memset(&header[0], 0, header.size());
#endif

    magicValue = peerMagicNumber;
    seqNo = rawSequenceNumber;

    str->write(&header[0], header.size());
  }
  
  writeData(str);
}


template <typename T, unsigned DIM, unsigned FLAGS_DIM> inline SampleData<T,DIM,FLAGS_DIM>::SampleData(const ExtentList &extents, const FlagsExtentList &flagsExtents, Allocator &allocator)
:
  samples(extents, alignment, allocator),
  flags(flagsExtents) // e.g., for FilteredData [nrChannels][nrStations], sparse dimension [nrSamplesPerIntegration]

  //itsHaveWarnedLittleEndian(false)
{ 
}


template <typename T, unsigned DIM, unsigned FLAGS_DIM> inline void SampleData<T,DIM,FLAGS_DIM>::readData(Stream *str)
{
  str->read(samples.origin(), samples.num_elements() * sizeof(T));
}


template <typename T, unsigned DIM, unsigned FLAGS_DIM> inline void SampleData<T,DIM,FLAGS_DIM>::writeData(Stream *str)
{
  str->write(samples.origin(), samples.num_elements() * sizeof(T));
}

} // namespace RTCP
} // namespace LOFAR

#endif

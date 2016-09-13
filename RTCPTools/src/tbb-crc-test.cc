/* tbb-crc-test.cpp
 * Author: Alexander S. van Amesfoort, ASTRON
 * with code based on Python crc routines received from Gijs Schoonderbeek, ASTRON
 * Last-modified: July 2012
 * build: g++ -Wall -o tbb-crc-test tbb-crc-test.cc
 */

#include <stdint.h>
#include <string.h>
#include <endian.h>
#if __BYTE_ORDER != __BIG_ENDIAN && __BYTE_ORDER != __LITTLE_ENDIAN
#error Byte order is neither big endian nor little endian: not supported
#endif
#include <byteswap.h>

#include <iostream>
#include <fstream>

#include <boost/crc.hpp>

using namespace std;

struct TBB_Header {
	uint8_t stationId;
	uint8_t rspID;
	uint8_t rcuID;
	uint8_t sampleFreq;

	uint32_t seqNr;
	uint32_t time;

	union {
		uint32_t sampleNr;
		uint32_t bandsliceNr;
	};

	uint16_t nOfSamplesPerFrame;
	uint16_t nOfFreqBands;

	uint8_t bandSel[64];

	uint16_t spare;
	uint16_t crc16;
};

#define MAX_TRANSIENT_NSAMPLES		1298 // based on frames stored by TBB and (un)packing
#define DEFAULT_TRANSIENT_NSAMPLES	1024 // int16_t
#define DEFAULT_SPECTRAL_NSAMPLES	 487 // complex int16_t
struct TBB_Payload {
	// For transient data, we typically receive 1024 samples per frame.
	// uint32_t crc comes right after, so cannot easily declare it after data[], hence + 2.
	int16_t data[MAX_TRANSIENT_NSAMPLES + 2];
};

struct TBB_Frame {
	TBB_Header header;
	TBB_Payload payload;
};


// Same truncated polynomials as standard crc16 and crc32, but with initial_remainder=0, final_xor_value=0, reflected_input=false, reflected_remainder_output=false.
// The boost::crc_optimal<> declarations precompute lookup tables, so do not declare inside the checking routine.
static boost::crc_optimal<16, 0x8005    /*, 0, 0, false, false*/> crc16tbbgen; // instead of crc_16_type
//boost::crc_basic<16> crc16tbbgen(0x8005/*, 0, 0, false, false*/); // non-opt variant

static boost::crc_optimal<32, 0x04C11DB7/*, 0, 0, false, false*/> crc32tbbgen; // instead of crc_32_type
//boost::crc_basic<32> crc32gen(0x04C11DB7/*, 0, 0, false, false*/); // non-opt variant

/*
 * Assumes that the seqNr field in the TBB_Frame at buf has been zeroed.
 * Takes a ptr to a complete header. (Drop too small frames earlier.)
 */
static bool crc16tbb_boost(const TBB_Header* header) {
	crc16tbbgen.reset();

	/*
	 * The header checksum is done like the data, i.e. on 16 bit little endian blocks at a time.
	 * As with the data, both big and little endian CPUs need to byte swap.
	 */
	const int16_t* ptr = reinterpret_cast<const int16_t*>(header);
	size_t i;
	for (i = 0; i < (sizeof(*header) - sizeof(header->crc16)) / sizeof(int16_t); i++) {
		int16_t val = __bswap_16(ptr[i]);
		crc16tbbgen.process_bytes(&val, sizeof(int16_t));
	}

	// Byte swap the little endian checksum on big endian only.
	// It is also possible to process header->crc16 and see if checksum() equals 0.
	uint16_t crc16val = header->crc16;
#if __BYTE_ORDER == __BIG_ENDIAN
	crc16val = __bswap_16(crc16val);
#endif
	return crc16tbbgen.checksum() == crc16val;
}

/*
 * Note: The nsamples arg is without the space taken by the crc32 in payload. (Drop too small frames earlier.)
 */
static bool crc32tbb_boost(const TBB_Payload* payload, size_t nsamples) {
	crc32tbbgen.reset();

	/*
	 * Both little and big endian CPUs need to byte swap, because the data always arrives
	 * in little and the boost routines treat it as uint8_t[] (big).
	 */
	const int16_t* ptr = reinterpret_cast<const int16_t*>(payload->data);
	size_t i;
	for (i = 0; i < nsamples; i++) {
		int16_t val = __bswap_16(ptr[i]);
		crc32tbbgen.process_bytes(&val, sizeof(int16_t));
	}

	// Byte swap the little endian checksum on big endian only.
	// It is also possible to process crc32val and see if checksum() equals 0.
	uint32_t crc32val = *reinterpret_cast<const uint32_t*>(&ptr[nsamples]);
#if __BYTE_ORDER == __BIG_ENDIAN
	crc32val = __bswap_32(crc32val);
#endif
	return crc32tbbgen.checksum() == crc32val;
}

#if __BYTE_ORDER != __LITTLE_ENDIAN
#warning Original crc routines were only developed for little endian. Skipping those.
#else
/*
 * This code is translated from the Python ref/test code received from Gijs Schoonderbeek.
 * It assumes that the seqNr field (buf[1]) has been zeroed.
 * Do not call this function with len < 1; reject too small headers earlier.
 */
static uint16_t crc16tbb(const uint16_t* buf, size_t len) {
	uint16_t CRC            = 0;
	const uint32_t CRC_poly = 0x18005;
	const uint16_t bits     = 16;
	uint32_t data           = 0;
	const uint32_t CRCDIV   = (CRC_poly & 0x7fffffff) << 15;

	data = (buf[0] & 0x7fffffff) << 16;
	for (uint32_t i = 1; i < len; i++) {
		data += buf[i];
		for (uint16_t j = 0; j < bits; j++) {
			if ((data & 0x80000000) != 0) {
				data = data ^ CRCDIV;
			}
			data = data & 0x7fffffff;
			data = data << 1;
		}
	}
	CRC = data >> 16;
	return CRC;
}

/*
 * This code is translated from the Python ref/test code received from Gijs Schoonderbeek.
 * It computes a 32 bit result, 16 bits at a time.
 * Do not call this function with len < 2; reject too small payloads earlier.
 */
static uint32_t crc32tbb(const uint16_t* buf, size_t len) {
	uint32_t CRC            = 0;
	const uint64_t CRC_poly = 0x104C11DB7ULL;
	const uint16_t bits     = 16;
	uint64_t data           = 0;
	const uint64_t CRCDIV   = (CRC_poly & 0x7fffffffffffULL) << 15;

	data = buf[0];
	data = data & 0x7fffffffffffULL;
	data = data << 16;
	data = data + buf[1];
	data = data & 0x7fffffffffffULL;
	data = data << 16;
	uint32_t i = 2;
	for ( ; i < len-2; i++) {
		data = data + buf[i];
		for (uint32_t j = 0; j < bits; j++) {
			if (data & 0x800000000000ULL) {
				data = data ^ CRCDIV;
			}
			data = data & 0x7fffffffffffULL;
			data = data << 1;
		}
	}

	// Do the 32 bit checksum separately.
	// Process the two 16 bit halves in reverse order, but keep the i < len cond.
	for (buf += 1; i < len; i++, buf -= 2) {
		data = data + buf[i];
		for (uint32_t j = 0; j < bits; j++) {
			if (data & 0x800000000000ULL) {
				data = data ^ CRCDIV;
			}
			data = data & 0x7fffffffffffULL;
			data = data << 1;
		}
	}

	CRC = (uint32_t)(data >> 16);
	return CRC;
}
#endif

static int verify_crc(TBB_Frame& frame, size_t frameSize) {
	int err = 0;

	// Zero sequence number field before verification.
	// It is set by TBB after the checksum has been computed. We do not need it later.
	frame.header.seqNr = 0;

#if __BYTE_ORDER == __LITTLE_ENDIAN
	uint16_t headercrc = crc16tbb( reinterpret_cast<uint16_t*>(&frame.header), sizeof(TBB_Header) / sizeof(uint16_t) );
	if (headercrc != 0) {
		cerr << "crc16tbb(): Incorrect header crc: " << hex << headercrc << endl;
		err = 1;
	}
#endif
	if (!crc16tbb_boost(&frame.header)) {
		cerr << "crc16tbb_boost(): Incorrect header crc" << endl;
		err = 1;
	}


#if __BYTE_ORDER == __LITTLE_ENDIAN
	uint32_t payloadcrc = crc32tbb( reinterpret_cast<uint16_t*>(frame.payload.data), ( frameSize - sizeof(TBB_Header) ) / sizeof(uint16_t) );
	if (payloadcrc != 0) {
		cerr << "crc32tbb(): Incorrect payload crc: " << hex << payloadcrc << endl;
		err = 1;
	}
#endif
	if (!crc32tbb_boost( &frame.payload, ( frameSize - sizeof(TBB_Header) - sizeof(uint32_t) ) / sizeof(int16_t) )) {
		cerr << "crc32tbb_boost(): Incorrect payload crc" << endl;
		err = 1;

#if 0 // this guessing doesn't work: the wrong crc32 is different every time, even on the same data
		TBB_Payload p;
		unsigned i;
		for (i = 0; i < 487; i++) {
			memcpy(&p, &frame.payload, i * 2 * sizeof(int16_t)); // data
			memcpy((char*)&p + i * 2 * sizeof(int16_t), (char*)(&frame.payload.data[2*487 + 2]) - sizeof(uint32_t), sizeof(uint32_t)); // crc32
			if (crc32tbb_boost(&p, 2 * i)) {
				cerr << "found it: i=" << i << endl;
				break;
			} else {
				cerr << "doesn't work either: " << i << endl;
			}
		}
#endif

	}

	return err;
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "Usage: " << argv[0] << " rawtbbframes.dat" << endl;
		return 1;
	}

	bool transient;
	ifstream iftype(argv[1], ios_base::binary);
	if (!iftype) {
		cerr << "Failed to open file " << argv[1] << endl;
		return 1;
	}
	TBB_Header header;
	iftype.read(reinterpret_cast<char*>(&header), sizeof header);
	if (!iftype) {
		cerr << "Failed to read first frame to determine transient or spectral mode" << endl;
		return 1;
	}
	iftype.close();
	transient = header.nOfFreqBands == 0;


	ifstream ifs(argv[1], ios_base::binary);
	if (!ifs) {
		cerr << "Failed to open file " << argv[1] << endl;
		return 1;
	}

	int err = 0;

	TBB_Frame frame;
	size_t frameSize;
	if (transient) {
		frameSize = sizeof(TBB_Header) + DEFAULT_TRANSIENT_NSAMPLES * sizeof(int16_t) + sizeof(uint32_t);
	} else { // spectral
		frameSize = sizeof(TBB_Header) + DEFAULT_SPECTRAL_NSAMPLES * 2 * sizeof(int16_t) + sizeof(uint32_t);
	}

	while (ifs.read(reinterpret_cast<char*>(&frame), frameSize)) {
		err |= verify_crc(frame, frameSize);
	}

	ifs.close();
	return err;
}


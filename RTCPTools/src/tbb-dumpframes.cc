/* tbb-dumpframes.cc
 * Author: Alexander S. van Amesfoort, ASTRON
 * Last-modified: Jun 2013
 * build: g++ -Wall -o tbb-dumpframes tbb-dumpframes.cc
 */

#include <stdint.h>
#include <cstdlib>
#include <cstring>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

struct TBB_Header {
	uint8_t stationID;	// Data source station identifier
	uint8_t rspID;		// Data source RSP board identifier
	uint8_t rcuID;		// Data source RCU board identifier
	uint8_t sampleFreq;	// Sample frequency in MHz of the RCU boards

	uint32_t seqNr;		// Used internally by TBB. Set to 0 by RSP (but written again before we receive it)
	uint32_t time;		// Time instance in seconds of the first sample in payload
	// The time field is relative, but if used as UNIX time, uint32_t will wrap at 06:28:15 UTC on 07 Feb 2106 (int32_t wraps at 03:14:08 UTC on 19 Jan 2038).

	union {
		// In transient mode indicates sample number of the first payload sample in current seconds interval.
		uint32_t sampleNr;

		// In spectral mode indicates frequency band and slice (transform block of 1024 samples) of first payload sample.
		uint32_t bandSliceNr; // bandNr[9:0] and sliceNr[31:10].
		// Avoid bit fields, (portable) compilation support is messy. Instead use mask and shift to decode.
#define TBB_BAND_NR_MASK	((1 << 10) - 1) 
#define TBB_SLICE_NR_SHIFT	10
	};

	uint16_t nOfSamplesPerFrame; // Total number of samples in the frame payload
	uint16_t nOfFreqBands;	// Number of frequency bands for each spectrum in spectral mode. Is set to 0 for transient mode.

	uint8_t bandSel[64];	// Each bit in the band selector field indicates whether the band with the bit index is present in the spectrum or not.

	uint16_t spare;		// For future use. Set to 0.
	uint16_t crc16;		// CRC16 over frame header, with seqNr set to 0.
};

void timeToStr(time_t t, char* out, size_t out_sz) {
        struct tm *tm = gmtime(&t);
        // Format: Mo, 15-06-2009 20:20:00
        strftime(out, out_sz, "%a, %d-%m-%Y %H:%M:%S", tm);
}

void printHeader(const TBB_Header& h) {
	cout << "Station ID:  " << (uint32_t)h.stationID << endl;
	cout << "RSP ID:      " << (uint32_t)h.rspID << endl;
	cout << "RCU ID:      " << (uint32_t)h.rcuID << endl;
	cout << "Sample Freq: " << (uint32_t)h.sampleFreq << endl;
	cout << "Seq Nr:      " << h.seqNr << endl;
	char buf[32];
	timeToStr(h.time, buf, 32);
	cout << "Time:        " << h.time << " (dd-mm-yyyy: " << buf << " UTC)" << endl;
	bool transient = h.nOfFreqBands == 0;
	if (transient) {
		cout << "Transient" << endl;
		cout << "Sample Nr:   " << h.sampleNr << endl;
	} else {
		cout << "Spectral" << endl;
		cout << "Band Nr:     " << (h.bandSliceNr & TBB_BAND_NR_MASK) << endl;
		cout << "Slice Nr:    " << (h.bandSliceNr >> TBB_SLICE_NR_SHIFT) << endl;
	}
	cout << "NSamples/fr: " << h.nOfSamplesPerFrame << endl;
	if (!transient) {
		cout << "NFreq Bands: " << h.nOfFreqBands << endl;

		bool anyBandsPresent = false;
		cout << "Band(s) present(?): ";
		for (unsigned i = 0; i < 64; ++i) {
			for (unsigned j = 8; j > 0; ) {
				--j;
				if (h.bandSel[i] & (1 << j)) {
					cout << 8 * i + (8-1-j) << " ";
					anyBandsPresent = true;
				}
			}
		}
		if (!anyBandsPresent) {
			cout << "Warning: Spectral data, but no band present!" << endl;
		} else {
			cout << endl;
		}
	}

	cout << "Spare (0):   " << h.spare << endl;
	cout << "crc16:       " << h.crc16 << endl;
}

void printPayload(const int16_t* payload, size_t payload_len) {
	size_t data_len = payload_len - sizeof(uint32_t) / sizeof(int16_t); // - crc32
	unsigned i;

	if (data_len == 1024) { // transient has 1024 samples + crc32
		for (i = 0; i < data_len; i++) {
			cout << payload[i] << " ";
		}
	} else { // spectral has up to 487 complex samples + crc32
		for (i = 0; i < data_len; i += 2) {
			cout << '(' << payload[i] << ' ' << payload[i+1] << ") "; // assumes data_len is even
		}
	}
	cout << endl;

	cout << "crc32:       " << reinterpret_cast<uint32_t*>(payload[i]) << endl;
}

void printFakeInput() {
	TBB_Header hdr0;

	hdr0.stationID = 1;
	hdr0.rspID = 2;
	hdr0.rcuID = 3;
	hdr0.sampleFreq = 200;
	hdr0.seqNr = 10000;
	hdr0.time = 1380240059;
	hdr0.bandSliceNr = (17 << 10) | 11; // sliceNr=17; bandNr is 11
	hdr0.nOfSamplesPerFrame = 487;
	hdr0.nOfFreqBands = 487/8 * 7 + 7; // 427, as set in the sb bitmap below

	// subband bitmap
	// I'm not 100% if the bits are populated from most to least significant...
	int i;
	for (i = 0; i < 487/8; i++)
		hdr0.bandSel[i] = 0x7f;
	hdr0.bandSel[i++] = 0xfe; // remaining 7 bits to cover all 487 meaningful bits
	for ( ; i < 64; i++)
		hdr0.bandSel[i] = 0;

	hdr0.spare = 0;
	hdr0.crc16 = 1;

	printHeader(hdr0);
}

int main(int argc, char* argv[]) {
	bool printData = false;
	bool fakeInput = false;
	const char* filename = "/dev/stdin";
	int nprinted = 8;

	cout << "Usage: " << argv[0] << " [-d] [-t] [data/tbbdata.raw] [nframes]" << endl;

	int argi = 1;
	if (argc > argi) {
		if (strcmp(argv[argi], "-d") == 0) {
			printData = true;
			argi += 1;
		}

		if (strcmp(argv[argi], "-t") == 0) {
			fakeInput = true;
			argi += 1;
		}

		if (argc > argi) {
			filename = argv[argi];
			argi += 1;
		}

		if (argc > argi) {
			nprinted = std::atoi(argv[argi]);
			argi += 1;
			if (nprinted < 0) {
				cerr << "Bad nframes argument" << endl;
				return 1;
			}
		}
	}


	if (fakeInput) {
		printFakeInput();
		exit(0);
	}

	ifstream ifs(filename);
	if (!ifs) {
		cerr << "Failed to open " << filename << endl;
		return 1;
	}

	cout << "Default frame size:" << " header=" << sizeof(TBB_Header) <<
		" transient=" << sizeof(TBB_Header) + 1024 * sizeof(int16_t) + sizeof(uint32_t) <<
		" spectral=" << sizeof(TBB_Header) + 487 * 2 * sizeof(int16_t) + sizeof(uint32_t) << endl << endl;

	int exit_status = 0;

	// This doesn't work directly with data from message-oriented streams like udp,
	// because header and payload need to be read using a single read() under linux.
	// We don't need that for dumping data from a file; buffers are separate here.
	TBB_Header h;
	int16_t* payload = NULL;
	for (int i = 0; i < nprinted; i++) {
		ifs.read(reinterpret_cast<char*>(&h), sizeof h);
		if (!ifs || static_cast<size_t>(ifs.gcount()) < sizeof h) {
			cerr << "Failed to read " << sizeof h << " frame header bytes from " << filename << endl;
			exit_status = 1;
			goto out;
		}

		printHeader(h);


		size_t payload_len = h.nOfSamplesPerFrame;
		if (h.nOfFreqBands != 0) {
			payload_len *= 2; // spectral has complex nrs, so 2 * int16_t
		}
		payload_len += sizeof(uint32_t) / sizeof(int16_t); // crc32
		if (payload == NULL) {
			// assume this is enough for all future frames; this program is for formatted frame dumps, not for the real thing anyway
			payload = new int16_t[payload_len]; // data + crc32
		}

		ifs.read(reinterpret_cast<char*>(payload), payload_len * sizeof(int16_t));
		if (!ifs) {
			cerr << "Failed to read " << payload_len * sizeof(int16_t) << " frame payload from " << filename << endl;
			exit_status = 1;
			goto out;
		}
		if (printData) {
			printPayload(payload, payload_len);
		}

		cout << "----------------------------" << endl;
	}

out: // too lazy to use proper objects in this test prog, but avoid mem leaks.....
	delete[] payload;
	return exit_status;
}


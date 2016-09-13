#if !defined FORMAT_H
#define FORMAT_H

#define MAX_CORES	64
#define MAX_CHANNELS	 8


struct RequestPacket {
  enum {
    ZERO_COPY_READ,
    ZERO_COPY_WRITE,
    RESET
  }		 type;
  unsigned short rank;
  unsigned short core;
  unsigned short rankInPSet; // logical; not the incomprehensible BG/P number!
  unsigned short channel;
  unsigned	 size;
  char		 messageHead[240];
};

#endif

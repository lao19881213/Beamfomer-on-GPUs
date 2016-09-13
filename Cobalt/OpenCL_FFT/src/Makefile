SRCS = fft_execute.cpp fft_setup.cpp fft_kernelstring.cpp
HEADERS = procs.h fft_internal.h fft_base_kernels.h clFFT.h
LIBRARY = libOpenCL_FFT.a
COMPILERFLAGS = -c -g -Wall -Werror -O3 -I..
CFLAGS = $(COMPILERFLAGS) ${RC_CFLAGS} ${USE_ATF}
CC = g++

OBJECTS = fft_execute.o fft_setup.o fft_kernelstring.o
TARGETOBJECT =
all: $(LIBRARY)

$(OBJECTS): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(SRCS)

$(LIBRARY): $(OBJECTS)
	ar r $@ $^

clean:
	rm -f $(TARGET) $(OBJECTS)

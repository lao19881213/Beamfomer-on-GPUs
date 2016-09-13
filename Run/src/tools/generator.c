#define  _BSD_SOURCE

#include "common.h"

#include <byteswap.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#define MAX_SOCKETS	64


double   rate		   = 195312.5;
unsigned subbands	   = 61;
unsigned samples_per_frame = 16;
unsigned bits_per_sample   = 16;
char	 packet[9000];
unsigned message_size;
int	 sockets[MAX_SOCKETS];
unsigned nr_sockets	   = 0;
unsigned packets_sent[MAX_SOCKETS], skipped, errors[MAX_SOCKETS];
char     names[MAX_SOCKETS][64];


void *log_thread(void *arg)
{
  while (1) {
    unsigned socket_nr;

    sleep(1);

    for (socket_nr = 0; socket_nr < nr_sockets; socket_nr ++)
      if (packets_sent[socket_nr] > 0 || errors[socket_nr] > 0)  {
	fprintf(stderr, "sent %u packets to %s, skipped = %u, errors = %u\n", packets_sent[socket_nr], names[socket_nr], skipped, errors[socket_nr]);
	packets_sent[socket_nr] = errors[socket_nr] = 0; // ignore race
      }

    skipped = 0;
  }

  return 0;
}


void send_packet(unsigned socket_nr, unsigned seconds, unsigned fraction)
{
#if defined __BIG_ENDIAN__
  * (int *) (packet +  8) = __bswap_32(seconds);
  * (int *) (packet + 12) = __bswap_32(fraction);
#else
  * (int *) (packet +  8) = seconds;
  * (int *) (packet + 12) = fraction;
#endif

  ++ packets_sent[socket_nr];

#if 1
  unsigned bytes_written;

  for (bytes_written = 0; bytes_written < message_size;) {
    ssize_t retval = write(sockets[socket_nr], packet + bytes_written, message_size - bytes_written);

    if (retval < 0) {
      ++ errors[socket_nr];
      perror("write");
      sleep(1);
      break;
    } else {
      bytes_written += retval;
    }
  }
#endif
}


void parse_args(int argc, char **argv)
{
  if (argc == 1) {
    fprintf(stderr, "usage: %s [-f frequency (default 195312.5)] [-s subbands (default 61)] [-t times_per_frame (default 16)] [udp:ip:port | tcp:ip:port | file:name | null: | - ] ... \n", argv[0]);
    exit(1);
  }

  int arg;

  for (arg = 1; arg < argc && argv[arg][0] == '-'; arg ++)
    switch (argv[arg][1]) {
      case 'a': set_affinity(argument(&arg, argv));
		break;

      case 'b': bits_per_sample = atoi(argument(&arg, argv));
		break;

      case 'f': rate = atof(argument(&arg, argv));
		break;

      case 'r': set_real_time_priority();
		break;

      case 's': subbands = atoi(argument(&arg, argv));
		break;

      case 't': samples_per_frame = atoi(argument(&arg, argv));
		break;

      default : fprintf(stderr, "unrecognized option '%c'\n", argv[arg][1]);
		exit(1);
    }

  if (arg == argc)
    exit(0);

  enum proto proto;

  for (nr_sockets = 0; arg != argc && nr_sockets < MAX_SOCKETS; arg ++, nr_sockets ++)
    sockets[nr_sockets] = create_fd(argv[arg], 1, &proto, names[nr_sockets], sizeof names[nr_sockets]);

  if (arg != argc)
    fprintf(stderr, "Warning: too many sockets specified\n");
}


int main(int argc, char **argv)
{
  if_BGP_set_default_affinity();
  parse_args(argc, argv);
  message_size = 16 + samples_per_frame * subbands * bits_per_sample / 2;

  pthread_t thread;

  if (pthread_create(&thread, 0, log_thread, 0) != 0) {
    perror("pthread_create");
    exit(1);
  }

  unsigned clock_speed = 1024 * rate;
  struct timeval now;
  gettimeofday(&now, 0);
  unsigned long long packet_time = (now.tv_sec + now.tv_usec / 1e6) * rate;

  while (1) {
    packet_time += samples_per_frame;

    gettimeofday(&now, 0);
    unsigned long long now_us = 1000000ULL * now.tv_sec + now.tv_usec;
    unsigned long long pkt_us = 1000000ULL * (packet_time / rate);

    long long wait_us = pkt_us - now_us;

    if (wait_us > 10)
      usleep(wait_us);
    else if (pkt_us + 100000 < now_us) {
      unsigned skip = (unsigned long long) (-wait_us * 1e-6 * rate) / samples_per_frame;
      skipped += skip;
      packet_time += skip * samples_per_frame; // skip packets; keep modulo(samples_per_frame)
    }

    unsigned seconds  = 1024 * packet_time / clock_speed;
    unsigned fraction = 1024 * packet_time % clock_speed / 1024;
    unsigned socket_nr;

    for (socket_nr = 0; socket_nr < nr_sockets; socket_nr ++)
      send_packet(socket_nr, seconds, fraction);
  }

  return 0;
}

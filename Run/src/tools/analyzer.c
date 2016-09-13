#include "common.h"

#include <byteswap.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>


double   rate		   = 195312.5;
unsigned subbands	   = 61;
unsigned samples_per_frame = 16;
unsigned bits_per_sample   = 16;

char	 name[64];
unsigned message_size;
int	 sk;
unsigned packets_received, missed, out_of_order, errors, bad_timestamps;
unsigned clock_speed;
unsigned long long expected_time;


int receive_packet()
{
  char packet[9000];
  size_t bytes_received;

  for (bytes_received = 0; bytes_received < message_size;) {
    ssize_t retval = read(sk, packet + bytes_received, message_size - bytes_received);

    if (retval < 0) {
      ++ errors;
      perror("read");
      sleep(1);
    } else if (retval == 0) {
      return 0;
    } else {
      bytes_received += retval;
    }
  }

  ++ packets_received;

#if defined __BIG_ENDIAN__
  unsigned seconds  = __bswap_32(* (int *) (packet +  8));
  unsigned fraction = __bswap_32(* (int *) (packet + 12));
#else
  unsigned seconds  = * (int *) (packet +  8);
  unsigned fraction = * (int *) (packet + 12);
#endif

  if (seconds == 0xFFFFFFFF) {
    ++ bad_timestamps;
  } else {
    unsigned long long time = ((unsigned long long) seconds * clock_speed + 512) / 1024 + fraction;

    if (expected_time != time && expected_time != 0)
      missed += (time - expected_time) / samples_per_frame;

    if (time < expected_time)
      ++ out_of_order;

    expected_time = time + samples_per_frame;
  }

  return 1;
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
  sk = create_fd(argv[arg], 0, &proto, name, sizeof name);
}


void print_message()
{
  fprintf(stderr, "received %u packets from %s", packets_received, name);

  if (missed > 0)
    fprintf(stderr, ", missed = %u", missed);

  if (out_of_order > 0)
    fprintf(stderr, ", out of order = %u", out_of_order);

  if (errors > 0)
    fprintf(stderr, ", read errors = %u", errors);

  if (bad_timestamps > 0)
    fprintf(stderr, ", bad timestamps = %u", bad_timestamps);

  fputc('\n', stderr);

  packets_received = missed = out_of_order = errors = bad_timestamps = 0;
}


int main(int argc, char **argv)
{
  if_BGP_set_default_affinity();
  setvbuf(stderr, 0, _IOLBF, 0);
  parse_args(argc, argv);

  message_size = 16 + samples_per_frame * subbands * bits_per_sample / 2;
  clock_speed  = (unsigned) (1024 * rate);

  time_t last_time;

  while (receive_packet()) {
    time_t current_wtime = time(0);

    if (current_wtime != last_time) {
      last_time = current_wtime;
      print_message();
    }
  }

  print_message();
  return 0;
}

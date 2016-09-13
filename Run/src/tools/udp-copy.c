/* Copyright 2008, John W. Romein, Stichting ASTRON
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#define  _GNU_SOURCE
#include "common.h"

// allow opening >2GB files on 32-bit architectures
#define _FILE_OFFSET_BITS 64

#include <features.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/filter.h>
#include <netdb.h>
#include <netinet/in.h>
#include <assert.h>
#include <poll.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


enum proto input_proto, output_proto;
char	   source[64], destination[64];

int	   sk_in, sk_out;
size_t     nr_packets = 0, nr_bytes = 0;

struct speed {
  struct timeval current_time;
  struct timeval previous_time;
};

void init_speed( struct speed *s ) {
  gettimeofday(&s->current_time, NULL);
  s->previous_time = s->current_time;
}

void update_speed( struct speed *s, double *speedptr, const char **suffixptr, size_t bytes ) {
  gettimeofday(&s->current_time, NULL);

  double current = 1.0 * s->current_time.tv_sec + s->current_time.tv_usec / 1.0e6;
  double prev    = 1.0 * s->previous_time.tv_sec + s->previous_time.tv_usec / 1.0e6;

  double speed = current == prev ? 0.0 : 8.0 * bytes / (current - prev);
  const char *suffix;

  if( speed > 1000*1000*1000 ) {
    speed /= 1000*1000*1000;
    suffix = "Gbit/s";
  } else if( speed > 1000*1000 ) {
    speed /= 1000*1000;
    suffix = "Mbit/s";
  } else if( speed > 1000 ) {
    speed /= 1000;
    suffix = "Kbit/s";
  } else {
    suffix = "bit/s";
  }

  *speedptr = speed;
  *suffixptr = suffix;

  s->previous_time = s->current_time;
}


void *log_thread(void *arg)
{
  struct speed speed;
  double speedval;
  const char *suffix;

  init_speed(&speed);

  while (1) {
    sleep(1);

    size_t bytes = nr_bytes;
    size_t packets = nr_packets;

    nr_packets = nr_bytes = 0;

    update_speed(&speed, &speedval, &suffix, bytes);

    if (packets > 0) {
      if (input_proto == UDP || input_proto == Eth)
	fprintf(stderr, "copied %zu bytes (= %zu packets) from %s to %s (%.2f %s)\n", bytes, packets, source, destination, speedval, suffix);
      else
	fprintf(stderr, "copied %zu bytes from %s to %s (%.2f %s)\n", bytes, source, destination, speedval, suffix);
    }
  }

  return 0;
}


void init(int argc, char **argv)
{
  int arg;

  for (arg = 1; arg < argc && argv[arg][0] == '-' && argv[arg][1] != '\0'; arg ++)
    switch (argv[arg][1]) {
      case 'r': set_real_time_priority();
		break;
    }

  if (arg + 2 != argc) {
    fprintf(stderr, "Usage: \"%s [-r] src-addr dest-addr\", where -r sets RT priority and addr is [tcp:|udp:]ip-addr:port or [file:]filename\n", argv[0]);
    exit(1);
  }

  sk_in  = create_fd(argv[arg], 0, &input_proto, source, sizeof source);
  sk_out = create_fd(argv[arg + 1], 1, &output_proto, destination, sizeof destination);

  setlinebuf(stdout);
  if_BGP_set_default_affinity();
}


int main(int argc, char **argv)
{
  char	   buffer[1024 * 1024] __attribute__ ((aligned(16)));
  size_t   read_size, write_size;

  init(argc, argv);

#if defined USE_RING_BUFFER
  time_t   previous_time = 0, current_time;

  if (input_proto == Eth) {
    unsigned offset = 0;
    while (1) {
      void *frame = ((char *) ring_buffer + offset * 8192);
      struct tpacket_hdr *hdr = frame;

#if 1
      if (hdr->tp_status == TP_STATUS_KERNEL) {
	struct pollfd pfd;

	pfd.fd = sk_in;
	pfd.revents = 0;
	pfd.events = POLLIN|POLLERR;

	if (poll(&pfd, 1, -1) < 0)
	  perror("poll");
      }
#else
      while (* (volatile long *) &hdr->tp_status == TP_STATUS_KERNEL)
	;
#endif

      assert((hdr->tp_status & 1) == TP_STATUS_USER); // FIXME

      //printf("status = %d %d %d %d %d %d %d\n", hdr->tp_status, hdr->tp_len, hdr->tp_snaplen, hdr->tp_mac, hdr->tp_net, hdr->tp_sec, hdr->tp_usec);
      unsigned char *mac = (char *) frame + hdr->tp_mac;
      //printf("mac = %02hhx:%02hhx:%02hhx:%02hhx:%02hhx:%02hhx\n", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
      unsigned char *data = (char *) frame + hdr->tp_net;
      //printf("data =");
      //unsigned i;
      //for (i = 0; i < 48; i ++)
	//printf(" %hhx", ((unsigned char *) data)[i]);
      //printf("\n");

      if (write(sk_out, data, hdr->tp_snaplen) < hdr->tp_snaplen) {
	perror("write");
	sleep(1);
      } else {
	nr_bytes += hdr->tp_snaplen;
      }

      ++ nr_packets;

      if ((current_time = time(0)) != previous_time) {
	previous_time = current_time;

	fprintf(stderr, "ok: copied %u bytes (= %u packets) from %s to %s\n", nr_bytes, nr_packets, source, destination);
	nr_packets = nr_bytes = 0;
      }

      hdr->tp_status = TP_STATUS_KERNEL;

      if (++ offset == 1024)
	offset = 0;
    }
  }
#endif
  pthread_t thread;

  if (pthread_create(&thread, 0, log_thread, 0) != 0) {
    perror("pthread_create");
    exit(1);
  }

  size_t max_size = output_proto == UDP ? 8960 : sizeof buffer;

  while ((read_size = read(sk_in, buffer, max_size)) != 0) {
    if (read_size < 0) {
      perror("read");
      sleep(1);
    } else {
      while (read_size > 0) {
	if ((write_size = write(sk_out, buffer, read_size)) < 0) {
	  perror("write");
	  sleep(1);
	} else {
	  read_size -= write_size;
	  nr_bytes  += write_size;
	}
      }
    }

    ++ nr_packets;

  }

  return 0;
}

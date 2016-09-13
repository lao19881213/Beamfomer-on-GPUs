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


#undef USE_RING_BUFFER

#define  _GNU_SOURCE
#include "common.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined __linux__
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/filter.h>
#include <linux/version.h>

#define IOPRIO_BITS		(16)
#define IOPRIO_CLASS_SHIFT	(13)
#define IOPRIO_PRIO_MASK	((1UL << IOPRIO_CLASS_SHIFT) - 1)

#define IOPRIO_PRIO_CLASS(mask)	((mask) >> IOPRIO_CLASS_SHIFT)
#define IOPRIO_PRIO_DATA(mask)	((mask) & IOPRIO_PRIO_MASK)
#define IOPRIO_PRIO_VALUE(class, data)	(((class) << IOPRIO_CLASS_SHIFT) | data)

#define IOPRIO_WHO_PROCESS	1
#define IOPRIO_WHO_PGRP		2
#define IOPRIO_WHO_USER		3

#define IOPRIO_CLASS_NONE	0
#define IOPRIO_CLASS_RT		1
#define	IOPRIO_CLASS_BE		2
#define	IOPRIO_CLASS_IDLE	3
#else
#define KERNEL_VERSION(X,Y,Z) 0
#endif


static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


int create_IP_socket(const char *arg, int is_output, enum proto proto)
{
  char		     *colon;
  struct sockaddr_in sa;
  struct hostent     *host;
  int		     sk, old_sk, buffer_size = 8 * 1024 * 1024;
  unsigned short     port;
  
  if ((colon = strchr(arg, ':')) == 0) {
    fprintf(stderr, "badly formatted IP:PORT address");
    exit(1);
  }

  port = colon[1] != '\0' ? strtol(colon + 1, 0, 0) : 0;
  *colon = '\0';

  pthread_mutex_lock(&mutex);

  if ((host = gethostbyname(arg)) == 0) {
    perror("gethostbyname");
    exit(1);
  }

  pthread_mutex_unlock(&mutex);

  memset(&sa, 0, sizeof sa);
  sa.sin_family = AF_INET;
  sa.sin_port   = htons(port);
  memcpy(&sa.sin_addr, host->h_addr, host->h_length);

  if ((sk = socket(AF_INET, proto == UDP ? SOCK_DGRAM : SOCK_STREAM, proto == UDP ? IPPROTO_UDP : IPPROTO_TCP)) < 0) {
    perror("socket");
    exit(1);
  }

  if (is_output) {
    while (connect(sk, (struct sockaddr *) &sa, sizeof sa) < 0) {
      if (errno == ECONNREFUSED) {
	sleep(1);
      } else {
	perror("connect");
	exit(1);
      }
    }

#if 1
    if (setsockopt(sk, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof buffer_size) < 0)
      perror("setsockopt failed");
#endif
  } else {
    if (bind(sk, (struct sockaddr *) &sa, sizeof sa) < 0) {
      perror("bind");
      exit(1);
    }

    if (proto == TCP) {
      listen(sk, 5);
      old_sk = sk;

      if ((sk = accept(sk, 0, 0)) < 0) {
	perror("accept");
	exit(1);
      }
      
      close(old_sk);
    }

#if 0
    if (setsockopt(sk, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof buffer_size) < 0)
      perror("setsockopt failed");
#endif
  }

#if 0 && defined USE_RING_BUFFER
  struct tpacket_req req;

  req.tp_block_size = 16384;
  req.tp_block_nr   = 64;
  req.tp_frame_size = 8192;
  req.tp_frame_nr   = req.tp_block_size / req.tp_frame_size * req.tp_block_nr;

  if (setsockopt(sk, SOL_SOCKET, PACKET_RX_RING, &req, sizeof req) < 0) {
    perror("ring buffer setsockopt");
    exit(1);
  }
#endif

#if 0
  int *buf;

  if ((buf = mmap(0, req.tp_block_size * req.tp_block_nr, PROT_READ|PROT_WRITE, MAP_SHARED, sk, 0)) == MAP_FAILED) {
    perror("ring buffer mmap");
    exit(1);
  }

  printf("%d %d %d %d\n", buf[0], buf[1], buf[2], buf[3]);
#endif

  return sk;
}


int create_file(const char *arg, int is_output)
{
  int fd;

  if ((fd = open(arg, is_output ? O_CREAT | O_WRONLY : O_RDONLY, 0666)) < 0) {
    perror("opening input file");
    exit(1);
  }

  return fd;
}


int create_stdio(int is_output, enum proto proto)
{
  switch (proto) {
    case StdIn	: if (is_output) {
		    fprintf(stderr, "Cannot write to stdin\n");
		    exit(1);
		  }

		  return dup(0);

    case StdOut	: if (!is_output) {
		    fprintf(stderr, "Cannot read from stdout\n");
		    exit(1);
		  }

		  return dup(1);

    case StdErr	: if (!is_output) {
		    fprintf(stderr, "Cannot read from stdout\n");
		    exit(1);
		  }

		  return dup(2);
    default     : fprintf(stderr, "Cannot create stdio with proto %d\n", proto), exit(1);
  }
}


void read_mac(const char *arg, char mac[6])
{
  if (sscanf(arg, "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx", mac, mac + 1, mac + 2, mac + 3, mac + 4, mac + 5) != 6) {
    fprintf(stderr, "bad MAC address");
    exit(1);
  }
}


int create_raw_eth_socket(const char *desc, int is_output)
{
  char *copy = strdup(desc), *arg = copy;
  int sk;

  if ((sk = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) < 0) {
    perror("raw socket");
    exit(1);
  }

  char  src_mac[6], dst_mac[6], proto;
  short type, dst_port;
  int   has_src_mac = 0, has_dst_mac = 0, has_type = 0, has_proto = 0;
  int	has_dst_port = 0;

  if (strtok(arg, ",") != 0) {
    do {
      if (strncmp("src=", arg, 4) == 0) {
	read_mac(arg + 4, src_mac);
	has_src_mac = 1;
      } else if (strncmp("dst=", arg, 4) == 0) {
	read_mac(arg + 4, dst_mac);
	has_dst_mac = 1;
      } else if (strncmp("type=", arg, 5) == 0) {
	type = strtol(arg + 5, 0, 0);
	has_type = 1;
      } else if (strncmp("proto=", arg, 6) == 0) {
	proto = strtol(arg + 6, 0, 0);
	has_proto = 1;
      } else if (strncmp("dst_port=", arg, 9) == 0) {
	dst_port = strtol(arg + 9, 0, 0);
	has_dst_port = 1;
      } else {
	fprintf(stderr, "unknown option \"%s\"\n", arg);
	exit(1);
      }
    } while ((arg = strtok(0, ",")) != 0);

#define MAX_FILTER_LENGTH 16
    struct sock_filter mac_filter_insn[MAX_FILTER_LENGTH], *prog = mac_filter_insn + MAX_FILTER_LENGTH;
    unsigned jump_offset = 1;

    // FILTER IS CONSTRUCTED IN REVERSED ORDER!
    //
    * -- prog = (struct sock_filter) BPF_STMT(BPF_RET + BPF_K, 0); // wrong packet; ret 0
    * -- prog = (struct sock_filter) BPF_STMT(BPF_RET + BPF_K, 65535); // right packet; ret everything
    
    if (has_proto) {
      * -- prog = (struct sock_filter) BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, proto, 0 , jump_offset);
      * -- prog = (struct sock_filter) BPF_STMT(BPF_LD + BPF_B + BPF_ABS , 14 + 9);
      jump_offset += 2;
    }
    
    if (has_type) {
      * -- prog = (struct sock_filter) BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, htons(type), 0 , jump_offset);
      * -- prog = (struct sock_filter) BPF_STMT(BPF_LD + BPF_H + BPF_ABS , 12);
      jump_offset += 2;
    }

    if (has_dst_port) {
      * -- prog = (struct sock_filter) BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, htons(dst_port), 0 , jump_offset);
      * -- prog = (struct sock_filter) BPF_STMT(BPF_LD + BPF_H + BPF_ABS , 14 + 20 + 2);
      jump_offset += 2;
    }

    if (has_src_mac) {
      * -- prog = (struct sock_filter) BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, htonl(* (int *) (src_mac + 2)), 0 , jump_offset);
      * -- prog = (struct sock_filter) BPF_STMT(BPF_LD + BPF_W + BPF_ABS , 8);
      jump_offset += 2;
      * -- prog = (struct sock_filter) BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, htons(* (short *) src_mac), 0 , jump_offset);
      * -- prog = (struct sock_filter) BPF_STMT(BPF_LD + BPF_H + BPF_ABS , 6);
      jump_offset += 2;
    }

    if (has_dst_mac) {
      * -- prog = (struct sock_filter) BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, htons(* (short *) (dst_mac + 4)), 0 , jump_offset);
      * -- prog = (struct sock_filter) BPF_STMT(BPF_LD + BPF_H + BPF_ABS , 4);
      jump_offset += 2;
      * -- prog = (struct sock_filter) BPF_JUMP(BPF_JMP + BPF_JEQ + BPF_K, htonl(* (int *) dst_mac), 0 , jump_offset);
      * -- prog = (struct sock_filter) BPF_STMT(BPF_LD + BPF_W + BPF_ABS , 0);
      jump_offset += 2;
    }

    struct sock_fprog filter;
    memset(&filter, 0, sizeof(struct sock_fprog));
    filter.filter = prog;
    filter.len = mac_filter_insn + MAX_FILTER_LENGTH - prog;

    if (setsockopt(sk, SOL_SOCKET, SO_ATTACH_FILTER, &filter, sizeof(filter)) < 0) {
      perror("error creating filter");
      exit(1);
    }
  }

#if defined USE_RING_BUFFER
  struct tpacket_req req;

  req.tp_block_size = 131072;
  req.tp_block_nr   = 64;
  req.tp_frame_size = 8192;
  req.tp_frame_nr   = req.tp_block_size / req.tp_frame_size * req.tp_block_nr;

  if (setsockopt(sk, SOL_PACKET, PACKET_RX_RING, &req, sizeof req) < 0) {
    perror("ring buffer setsockopt");
    exit(1);
  }

  if ((ring_buffer = mmap(0, req.tp_block_size * req.tp_block_nr, PROT_READ|PROT_WRITE, MAP_SHARED, sk, 0)) == MAP_FAILED) {
    perror("ring buffer mmap");
    exit(1);
  }
#endif

  free(copy);
  return sk;
}


int create_fd(const char *arg, int is_output, enum proto *proto, char *name, size_t max_name_size)
{
  if (strncmp(arg, "udp:", 4) == 0 || strncmp(arg, "UDP:", 4) == 0) {
    *proto = UDP;
    arg += 4;
  } else if (strncmp(arg, "tcp:", 4) == 0 || strncmp(arg, "TCP:", 4) == 0) {
    *proto = TCP;
    arg += 4;
  } else if (strncmp(arg, "file:", 5) == 0) {
    *proto = File;
    arg += 5;
  } else if (strncmp(arg, "eth:", 4) == 0) {
    *proto = Eth;
    arg += 4;
  } else if (strcmp(arg, "null:") == 0) {
    *proto = File;
    arg = "/dev/null";
  } else if (strcmp(arg, "zero:") == 0) {
    *proto = File;
    arg = "/dev/zero";
  } else if (strcmp(arg, "stdin:") == 0) {
    *proto = StdIn;
    arg = "stdin";
  } else if (strcmp(arg, "stdout:") == 0) {
    *proto = StdOut;
    arg = "stdout";
  } else if (strcmp(arg, "stderr:") == 0) {
    *proto = StdErr;
    arg = "stderr";
  } else if (strchr(arg, ':') != 0) {
    *proto = UDP;
  } else if (strcmp(arg, "-") == 0) {
    *proto = is_output ? StdOut : StdIn;
    arg = is_output ? "stdout" : "stdin";
  } else {
    *proto = File;
  }

  if (name != 0)
    strncpy(name, arg, max_name_size);

  switch (*proto) {
    case UDP	:
    case TCP	: return create_IP_socket(arg, is_output, *proto);

    case File	: return create_file(arg, is_output);

    case Eth	: return create_raw_eth_socket(arg, is_output);

    case StdIn	:
    case StdOut:
    case StdErr	: return create_stdio(is_output, *proto);
    default     : fprintf(stderr, "Cannot create fd for unknown proto %d\n", *proto), exit(1);
  }
}


static int is_BGP()
{
  FILE *file = fopen("/proc/cpuinfo", "r");

  if (file != 0) {
    char buffer[256];

    while (fgets(buffer, sizeof buffer, file) != 0)
      if (strcmp(buffer, "cpu\t\t: 450 Blue Gene/P DD2\n") == 0) {
	fclose(file);
	return 1;
      }

    fclose(file);
  }

  return 0;
}


void set_affinity(const char *arg)
{
  cpu_set_t cpu_set;

  CPU_ZERO(&cpu_set);

  do {
    unsigned begin, end;

    switch (sscanf(arg, "%u-%u", &begin, &end)) {
      unsigned cpu;

      case 0 :	fprintf(stderr, "parse error in CPU affinity set\n");
		exit(1);

      case 1 :	CPU_SET(begin, &cpu_set);
		break;

      case 2 :	for (cpu = begin; cpu <= end; cpu ++)
		  CPU_SET(cpu, &cpu_set);

		break;
    }
  } while ((arg = strchr(arg, ',') + 1) != (char *) 1);

  if (sched_setaffinity(0, sizeof cpu_set, &cpu_set) != 0)
    perror("sched_setaffinity");
}


void if_BGP_set_default_affinity()
{
  if (is_BGP())
    set_affinity("1-3");
}


void set_real_time_priority()
{
#if defined __linux__ && LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,13)
  if (syscall(SYS_ioprio_set, IOPRIO_WHO_PROCESS, getpid(), IOPRIO_PRIO_VALUE(IOPRIO_CLASS_RT, 7)) < 0)
    perror("ioprio_set");
#else
  fprintf(stderr, "warning: kernel does not support IO priority\n");
#endif

#if defined __linux__ && LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,18)
  struct sched_param sp;
  sp.sched_priority = sched_get_priority_min(SCHED_RR);

  if (sched_setscheduler(0, SCHED_RR, &sp) < 0)
    perror("sched_setscheduler");
#else
  fprintf(stderr, "warning: kernel does not support RT scheduler\n");
#endif

#if defined __linux__
  if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0)
    perror("mlockall");
#else
  fprintf(stderr, "warning: kernel cannot lock application in memory\n");
#endif
}


char *argument(int *arg, char **argv)
{
  if (argv[*arg][2] != '\0')
    return &argv[*arg][2];

  if (argv[*arg + 1] != 0)
    return &argv[++ *arg][0];
    
  fprintf(stderr, "-%c requires argument\n", argv[*arg][1]);
  exit(1);
}


ssize_t readAll(int fd, void *ptr, size_t size)
{
  ssize_t retval;
  size_t received;

  for (received = 0; received < size; received += retval)
    if ((retval = read(fd, (char *) ptr + received, size - received)) <= 0)
      return retval;

  return size;
}


ssize_t writeAll(int fd, const void *ptr, size_t size)
{
  ssize_t retval;
  size_t sent;

  for (sent = 0; sent < size; sent += retval)
    if ((retval = write(fd, (const char *) ptr + sent, size - sent)) <= 0)
      return retval;

  return size;
}


#if 0
static unsigned offset = 0;

size_t get_packet()
{
  void *frame = ((char *) ring_buffer + offset * 8192);
  struct tpacket_hdr *hdr = frame;
  unsigned char *data = (char *) frame + hdr->tp_net;
  return hdr->tp_snaplen;
}

void packet_done()
{
  hdr->tp_status = TP_STATUS_KERNEL;

  if (++ offset == 1024)
    offset = 0;
}
#endif


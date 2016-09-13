#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>


volatile unsigned long long current_time;
double rate;
char packet[9000];
unsigned message_size;
int sk;
struct sockaddr_in sa;
unsigned packets_sent, errors;


void create_socket(const char *destination_ip, const char *destination_port)
{
  struct hostent *host;

  if ((sk = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
    perror("socket");
    exit(1);
  }

  if ((host = gethostbyname(destination_ip)) == 0) {
    perror("gethostbyname");
    exit(1);
  }

  memset(&sa, 0, sizeof sa);
  sa.sin_family       = AF_INET;
  sa.sin_port         = htons(atoi(destination_port));
  memcpy(&sa.sin_addr, host->h_addr, host->h_length);

  if (connect(sk, (struct sockaddr *) &sa, sizeof sa) < 0) {
    perror("connect");
    exit(1);
  }
}


void send_packet()
{
  // FIXME take care of endianness
  * (int *) (packet +  8) = current_time / rate;
  * (int *) (packet + 12) = fmod(current_time, rate);

  ++ packets_sent;

#if 1
  if (send(sk, packet, message_size, 0) < 0) {
    ++ errors;
    perror("send");
    sleep(1);
  }
#endif
}


int main(int argc, char **argv)
{
  time_t last_time;

  if (argc < 5 || argc > 6) {
    fprintf(stderr, "usage: %s dest-ip dest-port rate subbands [samples-per-frame]\n", argv[0]);
    exit(1);
  }

  create_socket(argv[1], argv[2]);
  unsigned subbands	     = atoi(argv[4]);
  unsigned samples_per_frame = argc == 6 ? atoi(argv[5]) : 16;
  message_size = 16 + samples_per_frame * subbands * 8;

  struct timeval tv;
  rate = atof(argv[3]);

  double interval = 1.0 / rate;
  tv.tv_sec  = interval;
  tv.tv_usec = 1e6 * (interval - floor(interval));

  printf("timer: %lu sec, %lu usec\n", tv.tv_sec, tv.tv_usec);

  while (1) {
    unsigned long long new_time;

    do {
      gettimeofday(&tv, 0);
      new_time = (tv.tv_sec /*- 7200*/ + tv.tv_usec / 1000000.0) * rate;
    } while (new_time < current_time + samples_per_frame);

    current_time = new_time / samples_per_frame * samples_per_frame;
    send_packet();

    time_t current_wtime = time(0);

    if (current_wtime != last_time) {
      last_time = current_wtime;
      fprintf(stderr, "sent %u packets to %s:%s, errors = %u\n", packets_sent, argv[1], argv[2], errors);
      packets_sent = errors = 0;
    }
  }

  return 0;
}

#if !defined STREAM_H
#define STREAM_H

#include <stddef.h>


enum proto { UDP, TCP, File, Eth, StdIn, StdOut, StdErr };

extern int create_fd(const char *desc, int is_output, enum proto *proto, char *name, size_t max_name_size);
void if_BGP_set_default_affinity();
void set_affinity(const char *arg);
void set_real_time_priority();
char *argument(int *arg, char **argv);
//ssize_t readAll(int fd, void *ptr, size_t size);
//ssize_t writeAll(int fd, const void *ptr, size_t size);

#endif

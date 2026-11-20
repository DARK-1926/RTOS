#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>
static double us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(){ struct rusage ru; char b[128]; int fd=open("/proc/loadavg",O_RDONLY); double t; int N=2000;
 t=us(); for(int i=0;i<N;i++) getrusage(RUSAGE_SELF,&ru); printf("getrusage %.2f us\n",(us()-t)/N);
 t=us(); for(int i=0;i<N;i++) getrusage(RUSAGE_THREAD,&ru); printf("getrusage(thread) %.2f us\n",(us()-t)/N);
 t=us(); for(int i=0;i<N;i++) pread(fd,b,127,0); printf("pread loadavg %.2f us\n",(us()-t)/N);
 int fs=open("/proc/stat",O_RDONLY); char big[8192]; t=us(); for(int i=0;i<200;i++) pread(fs,big,8191,0); printf("pread /proc/stat %.2f us\n",(us()-t)/200);
 int fp=open("/proc/pressure/cpu",O_RDONLY); printf("psi fd %d\n",fp); return 0; }

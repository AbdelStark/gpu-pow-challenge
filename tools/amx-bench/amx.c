// Intel AMX INT8 peak microbenchmark (register-resident tiles, no memory traffic).
// Build: gcc -O2 -mamx-tile -mamx-int8 -pthread tools/amx-bench/amx.c -o build/amx
// Run:   build/amx <threads>   -> INT8 GOPS (2 ops per multiply-accumulate)
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <stdlib.h>
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18
typedef struct { uint8_t palette, start; uint8_t r[14]; uint16_t colsb[16]; uint8_t rows[16]; } cfg_t;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void* run(void* arg){
  long iters=(long)arg;
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) { perror("prctl"); return 0; }
  cfg_t c; memset(&c,0,sizeof c); c.palette=1;
  for(int i=0;i<8;i++){c.colsb[i]=64;c.rows[i]=16;}
  _tile_loadconfig(&c);
  static int8_t a[16*64], b[16*64]; static int32_t o[16*16];
  for(int i=0;i<1024;i++){a[i]=(int8_t)(i*7);b[i]=(int8_t)(i*13);}
  _tile_loadd(4,a,64);_tile_loadd(5,a,64);_tile_loadd(6,b,64);_tile_loadd(7,b,64);
  _tile_zero(0);_tile_zero(1);_tile_zero(2);_tile_zero(3);
  for(long i=0;i<iters;i++){ _tile_dpbssd(0,4,6); _tile_dpbssd(1,4,7); _tile_dpbssd(2,5,6); _tile_dpbssd(3,5,7); }
  _tile_stored(0,o,64); _tile_release();
  return (void*)(intptr_t)o[0];
}
int main(int argc,char**argv){
  int th=argc>1?atoi(argv[1]):1; long iters=20000000/ (th>0?1:1);
  pthread_t t[64]; double t0=now();
  for(int i=0;i<th;i++) pthread_create(&t[i],0,run,(void*)iters);
  for(int i=0;i<th;i++) pthread_join(t[i],0);
  double dt=now()-t0; double ops=(double)th*iters*4*16*16*64*2;
  printf("threads=%d  %.1f GOPS total  %.1f GOPS/thread\n",th,ops/dt/1e9,ops/dt/1e9/th);
}

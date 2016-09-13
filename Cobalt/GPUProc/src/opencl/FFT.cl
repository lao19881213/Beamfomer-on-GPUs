// Run kernel fft0 with global dim = {64*BatchSize}, local dim={64}
#ifndef M_PI
//#define M_PI 0x1.921fb54442d18p+1
#define M_PI 3.1415926536f
#endif
#define complexMul(a,b) ((float2)(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)))
#define conj(a) ((float2)((a).x, -(a).y))
#define conjTransp(a) ((float2)(-(a).y, (a).x))

#define fftKernel2(a,dir) \
  { \
    float2 c = (a)[0];    \
    (a)[0] = c + (a)[1];  \
    (a)[1] = c - (a)[1];  \
  }

#define fftKernel2S(d1,d2,dir) \
  { \
    float2 c = (d1);   \
    (d1) = c + (d2);   \
    (d2) = c - (d2);   \
  }

#define fftKernel4(a,dir) \
  { \
    fftKernel2S((a)[0], (a)[2], dir); \
    fftKernel2S((a)[1], (a)[3], dir); \
    fftKernel2S((a)[0], (a)[1], dir); \
    (a)[3] = (float2)(dir) * (conjTransp((a)[3])); \
    fftKernel2S((a)[2], (a)[3], dir); \
    float2 c = (a)[1]; \
    (a)[1] = (a)[2]; \
    (a)[2] = c; \
  }

#define fftKernel4s(a0,a1,a2,a3,dir) \
  { \
    fftKernel2S((a0), (a2), dir); \
    fftKernel2S((a1), (a3), dir); \
    fftKernel2S((a0), (a1), dir); \
    (a3) = (float2)(dir) * (conjTransp((a3))); \
    fftKernel2S((a2), (a3), dir); \
    float2 c = (a1); \
    (a1) = (a2); \
    (a2) = c; \
  }

#define bitreverse8(a) \
  { \
    float2 c; \
    c = (a)[1]; \
    (a)[1] = (a)[4]; \
    (a)[4] = c; \
    c = (a)[3]; \
    (a)[3] = (a)[6]; \
    (a)[6] = c; \
  }

#define fftKernel8(a,dir) \
  { \
    const float2 w1 = (float2)(0x1.6a09e6p-1f,  dir * 0x1.6a09e6p-1f);  \
    const float2 w3 = (float2)(-0x1.6a09e6p-1f, dir * 0x1.6a09e6p-1f);  \
    float2 c; \
    fftKernel2S((a)[0], (a)[4], dir); \
    fftKernel2S((a)[1], (a)[5], dir); \
    fftKernel2S((a)[2], (a)[6], dir); \
    fftKernel2S((a)[3], (a)[7], dir); \
    (a)[5] = complexMul(w1, (a)[5]); \
    (a)[6] = (float2)(dir) * (conjTransp((a)[6])); \
    (a)[7] = complexMul(w3, (a)[7]); \
    fftKernel2S((a)[0], (a)[2], dir); \
    fftKernel2S((a)[1], (a)[3], dir); \
    fftKernel2S((a)[4], (a)[6], dir); \
    fftKernel2S((a)[5], (a)[7], dir); \
    (a)[3] = (float2)(dir) * (conjTransp((a)[3])); \
    (a)[7] = (float2)(dir) * (conjTransp((a)[7])); \
    fftKernel2S((a)[0], (a)[1], dir); \
    fftKernel2S((a)[2], (a)[3], dir); \
    fftKernel2S((a)[4], (a)[5], dir); \
    fftKernel2S((a)[6], (a)[7], dir); \
    bitreverse8((a)); \
  }

#define bitreverse4x4(a) \
  { \
    float2 c; \
    c = (a)[1];  (a)[1] = (a)[4];  (a)[4] = c; \
    c = (a)[2];  (a)[2] = (a)[8];  (a)[8] = c; \
    c = (a)[3];  (a)[3] = (a)[12]; (a)[12] = c; \
    c = (a)[6];  (a)[6] = (a)[9];  (a)[9] = c; \
    c = (a)[7];  (a)[7] = (a)[13]; (a)[13] = c; \
    c = (a)[11]; (a)[11] = (a)[14]; (a)[14] = c; \
  }

#define fftKernel16(a,dir) \
  { \
    const float w0 = 0x1.d906bcp-1f; \
    const float w1 = 0x1.87de2ap-2f; \
    const float w2 = 0x1.6a09e6p-1f; \
    fftKernel4s((a)[0], (a)[4], (a)[8],  (a)[12], dir); \
    fftKernel4s((a)[1], (a)[5], (a)[9],  (a)[13], dir); \
    fftKernel4s((a)[2], (a)[6], (a)[10], (a)[14], dir); \
    fftKernel4s((a)[3], (a)[7], (a)[11], (a)[15], dir); \
    (a)[5] = complexMul((a)[5], (float2)(w0, dir * w1)); \
    (a)[6] = complexMul((a)[6], (float2)(w2, dir * w2)); \
    (a)[7] = complexMul((a)[7], (float2)(w1, dir * w0)); \
    (a)[9] = complexMul((a)[9], (float2)(w2, dir * w2)); \
    (a)[10] = (float2)(dir) * (conjTransp((a)[10])); \
    (a)[11] = complexMul((a)[11], (float2)(-w2, dir * w2)); \
    (a)[13] = complexMul((a)[13], (float2)(w1, dir * w0)); \
    (a)[14] = complexMul((a)[14], (float2)(-w2, dir * w2)); \
    (a)[15] = complexMul((a)[15], (float2)(-w0, dir * -w1)); \
    fftKernel4((a), dir); \
    fftKernel4((a) + 4, dir); \
    fftKernel4((a) + 8, dir); \
    fftKernel4((a) + 12, dir); \
    bitreverse4x4((a)); \
  }

#define bitreverse32(a) \
  { \
    float2 c1, c2; \
    c1 = (a)[2];   (a)[2] = (a)[1];   c2 = (a)[4];   (a)[4] = c1;   c1 = (a)[8];   (a)[8] = c2;    c2 = (a)[16];  (a)[16] = c1;   (a)[1] = c2; \
    c1 = (a)[6];   (a)[6] = (a)[3];   c2 = (a)[12];  (a)[12] = c1;  c1 = (a)[24];  (a)[24] = c2;   c2 = (a)[17];  (a)[17] = c1;   (a)[3] = c2; \
    c1 = (a)[10];  (a)[10] = (a)[5];  c2 = (a)[20];  (a)[20] = c1;  c1 = (a)[9];   (a)[9] = c2;    c2 = (a)[18];  (a)[18] = c1;   (a)[5] = c2; \
    c1 = (a)[14];  (a)[14] = (a)[7];  c2 = (a)[28];  (a)[28] = c1;  c1 = (a)[25];  (a)[25] = c2;   c2 = (a)[19];  (a)[19] = c1;   (a)[7] = c2; \
    c1 = (a)[22];  (a)[22] = (a)[11]; c2 = (a)[13];  (a)[13] = c1;  c1 = (a)[26];  (a)[26] = c2;   c2 = (a)[21];  (a)[21] = c1;   (a)[11] = c2; \
    c1 = (a)[30];  (a)[30] = (a)[15]; c2 = (a)[29];  (a)[29] = c1;  c1 = (a)[27];  (a)[27] = c2;   c2 = (a)[23];  (a)[23] = c1;   (a)[15] = c2; \
  }

#define fftKernel32(a,dir) \
  { \
    fftKernel2S((a)[0],  (a)[16], dir); \
    fftKernel2S((a)[1],  (a)[17], dir); \
    fftKernel2S((a)[2],  (a)[18], dir); \
    fftKernel2S((a)[3],  (a)[19], dir); \
    fftKernel2S((a)[4],  (a)[20], dir); \
    fftKernel2S((a)[5],  (a)[21], dir); \
    fftKernel2S((a)[6],  (a)[22], dir); \
    fftKernel2S((a)[7],  (a)[23], dir); \
    fftKernel2S((a)[8],  (a)[24], dir); \
    fftKernel2S((a)[9],  (a)[25], dir); \
    fftKernel2S((a)[10], (a)[26], dir); \
    fftKernel2S((a)[11], (a)[27], dir); \
    fftKernel2S((a)[12], (a)[28], dir); \
    fftKernel2S((a)[13], (a)[29], dir); \
    fftKernel2S((a)[14], (a)[30], dir); \
    fftKernel2S((a)[15], (a)[31], dir); \
    (a)[17] = complexMul((a)[17], (float2)(0x1.f6297cp-1f, dir * 0x1.8f8b84p-3f)); \
    (a)[18] = complexMul((a)[18], (float2)(0x1.d906bcp-1f, dir * 0x1.87de2ap-2f)); \
    (a)[19] = complexMul((a)[19], (float2)(0x1.a9b662p-1f, dir * 0x1.1c73b4p-1f)); \
    (a)[20] = complexMul((a)[20], (float2)(0x1.6a09e6p-1f, dir * 0x1.6a09e6p-1f)); \
    (a)[21] = complexMul((a)[21], (float2)(0x1.1c73b4p-1f, dir * 0x1.a9b662p-1f)); \
    (a)[22] = complexMul((a)[22], (float2)(0x1.87de2ap-2f, dir * 0x1.d906bcp-1f)); \
    (a)[23] = complexMul((a)[23], (float2)(0x1.8f8b84p-3f, dir * 0x1.f6297cp-1f)); \
    (a)[24] = complexMul((a)[24], (float2)(0x0p+0f, dir * 0x1p+0f)); \
    (a)[25] = complexMul((a)[25], (float2)(-0x1.8f8b84p-3f, dir * 0x1.f6297cp-1f)); \
    (a)[26] = complexMul((a)[26], (float2)(-0x1.87de2ap-2f, dir * 0x1.d906bcp-1f)); \
    (a)[27] = complexMul((a)[27], (float2)(-0x1.1c73b4p-1f, dir * 0x1.a9b662p-1f)); \
    (a)[28] = complexMul((a)[28], (float2)(-0x1.6a09e6p-1f, dir * 0x1.6a09e6p-1f)); \
    (a)[29] = complexMul((a)[29], (float2)(-0x1.a9b662p-1f, dir * 0x1.1c73b4p-1f)); \
    (a)[30] = complexMul((a)[30], (float2)(-0x1.d906bcp-1f, dir * 0x1.87de2ap-2f)); \
    (a)[31] = complexMul((a)[31], (float2)(-0x1.f6297cp-1f, dir * 0x1.8f8b84p-3f)); \
    fftKernel16((a), dir); \
    fftKernel16((a) + 16, dir); \
    bitreverse32((a)); \
  }

__kernel void \
  clFFT_1DTwistInterleaved(__global float2 *in, unsigned int startRow, unsigned int numCols, unsigned int N, unsigned int numRowsToProcess, int dir) \
  { \
  float2 a, w; \
  float ang; \
  unsigned int j; \
  unsigned int i = get_global_id(0); \
  unsigned int startIndex = i; \
         \
  if(i < numCols) \
  { \
    for(j = 0; j < numRowsToProcess; j++) \
    { \
      a = in[startIndex]; \
      ang = 2.0f * M_PI * dir * i * (startRow + j) / N; \
      w = (float2)(native_cos(ang), native_sin(ang)); \
      a = complexMul(a, w); \
      in[startIndex] = a; \
      startIndex += numCols; \
    } \
  }        \
  } \
  __kernel void fft0(__global float2 *in_out)
{
  const int dir = -1;
  __local float2 sMem[4][272];
  int i, j;
  float ang, angf;
  __local float2 *lMemStore, *lMemLoad;
  float2 a0, a1, a2, a3;
  int offset = (get_group_id(0) * 4 + get_local_id(1)) * 256 + get_local_id(0);
  in_out += offset;
  a0 = in_out[0];
  a1 = in_out[64];
  a2 = in_out[128];
  a3 = in_out[192];
  fftKernel4s(a0, a1, a2, a3, dir);
  angf = (float) get_local_id(0);
  ang = dir * ( 2.0f * M_PI * 1.0f / 256.0f ) * angf;
  float2 w0 = (float2)(native_cos(ang), native_sin(ang));
  ang = dir * ( 2.0f * M_PI * 2.0f / 256.0f ) * angf;
  float2 w1 = (float2)(native_cos(ang), native_sin(ang));
  ang = dir * ( 2.0f * M_PI * 3.0f / 256.0f ) * angf;
  float2 w2 = (float2)(native_cos(ang), native_sin(ang));
  a1 = complexMul(a1, w0);
  a2 = complexMul(a2, w1);
  a3 = complexMul(a3, w2);
  lMemStore = &sMem[get_local_id(1)][get_local_id(0)];
  j = get_local_id(0) & 3;
  i = get_local_id(0) >> 2;
  lMemLoad = &sMem[get_local_id(1)][j * 68 + i];
  lMemStore[0] = a0;
  lMemStore[68] = a1;
  lMemStore[136] = a2;
  lMemStore[204] = a3;
  barrier(CLK_LOCAL_MEM_FENCE);
  a0 = lMemLoad[0];
  a1 = lMemLoad[16];
  a2 = lMemLoad[32];
  a3 = lMemLoad[48];
  barrier(CLK_LOCAL_MEM_FENCE);
  fftKernel4s(a0, a1, a2, a3, dir);
  angf = (float) (get_local_id(0) >> 2);
  ang = dir * ( 2.0f * M_PI * 1.0f / 64.0f ) * angf;
  float2 w3 = (float2)(native_cos(ang), native_sin(ang));
  ang = dir * ( 2.0f * M_PI * 2.0f / 64.0f ) * angf;
  float2 w4 = (float2)(native_cos(ang), native_sin(ang));
  ang = dir * ( 2.0f * M_PI * 3.0f / 64.0f ) * angf;
  float2 w5 = (float2)(native_cos(ang), native_sin(ang));
  a1 = complexMul(a1, w3);
  a2 = complexMul(a2, w4);
  a3 = complexMul(a3, w5);
  j = (get_local_id(0) & 15) >> 2;
  i = (get_local_id(0) >> 4) * 4 + (get_local_id(0) & 3);
  lMemLoad = &sMem[get_local_id(1)][j * 68 + i];
  lMemStore[0] = a0;
  lMemStore[68] = a1;
  lMemStore[136] = a2;
  lMemStore[204] = a3;
  barrier(CLK_LOCAL_MEM_FENCE);
  a0 = lMemLoad[0];
  a1 = lMemLoad[16];
  a2 = lMemLoad[32];
  a3 = lMemLoad[48];
  barrier(CLK_LOCAL_MEM_FENCE);
  fftKernel4s(a0, a1, a2, a3, dir);
  angf = (float) (get_local_id(0) >> 4);
  ang = dir * ( 2.0f * M_PI * 1.0f / 16.0f ) * angf;
  float2 w6 = (float2)(native_cos(ang), native_sin(ang));
  ang = dir * ( 2.0f * M_PI * 2.0f / 16.0f ) * angf;
  float2 w7 = (float2)(native_cos(ang), native_sin(ang));
  ang = dir * ( 2.0f * M_PI * 3.0f / 16.0f ) * angf;
  float2 w8 = (float2)(native_cos(ang), native_sin(ang));
  a1 = complexMul(a1, w6);
  a2 = complexMul(a2, w7);
  a3 = complexMul(a3, w8);
  j = get_local_id(0) >> 4;
  i = get_local_id(0) & 15;
  lMemLoad = &sMem[get_local_id(1)][j * 64 + i];
  lMemStore[0] = a0;
  lMemStore[64] = a1;
  lMemStore[128] = a2;
  lMemStore[192] = a3;
  barrier(CLK_LOCAL_MEM_FENCE);
  a0 = lMemLoad[0];
  a1 = lMemLoad[16];
  a2 = lMemLoad[32];
  a3 = lMemLoad[48];
  fftKernel4s(a0, a1, a2, a3, dir);
  in_out[0] = a0;
  in_out[64] = a1;
  in_out[128] = a2;
  in_out[192] = a3;
}


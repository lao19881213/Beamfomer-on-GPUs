

// Copyright (C) 2010-2012 Advanced Micro Devices, Inc. All Rights Reserved.


__constant float2 twiddles[7] = {
  (float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
  (float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
  (float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
  (float2)(1.0000000000000000000000000000000000f, -0.0000000000000000000000000000000000f),
  (float2)(0.7071067811865475727373109293694142f, -0.7071067811865474617150084668537602f),
  (float2)(0.0000000000000000612323399573676604f, -1.0000000000000000000000000000000000f),
  (float2)(-0.7071067811865474617150084668537602f, -0.7071067811865475727373109293694142f),
};


#define fvect2 float2

#define C8Q  0.70710678118654752440084436210485f
#define C5QA 0.30901699437494742410229341718282f
#define C5QB 0.95105651629515357211643933337938f
#define C5QC 0.50000000000000000000000000000000f
#define C5QD 0.58778525229247312916870595463907f
#define C5QE 0.80901699437494742410229341718282f
#define C3QA 0.50000000000000000000000000000000f
#define C3QB 0.86602540378443864676372317075294f

__attribute__((always_inline)) void
FwdRad2B1(float2 *R0, float2 *R1)
{

  float2 T;

  (*R1) = (*R0) - (*R1);
  (*R0) = 2.0f * (*R0) - (*R1);


}

__attribute__((always_inline)) void
InvRad2B1(float2 *R0, float2 *R1)
{

  float2 T;

  (*R1) = (*R0) - (*R1);
  (*R0) = 2.0f * (*R0) - (*R1);


}

__attribute__((always_inline)) void
FwdRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

  float2 T;

  (*R1) = (*R0) - (*R1);
  (*R0) = 2.0f * (*R0) - (*R1);
  (*R3) = (*R2) - (*R3);
  (*R2) = 2.0f * (*R2) - (*R3);

  (*R2) = (*R0) - (*R2);
  (*R0) = 2.0f * (*R0) - (*R2);
  (*R3) = (*R1) + (fvect2)(-(*R3).y, (*R3).x);
  (*R1) = 2.0f * (*R1) - (*R3);

  T = (*R1);
  (*R1) = (*R2);
  (*R2) = T;

}

__attribute__((always_inline)) void
InvRad4B1(float2 *R0, float2 *R2, float2 *R1, float2 *R3)
{

  float2 T;

  (*R1) = (*R0) - (*R1);
  (*R0) = 2.0f * (*R0) - (*R1);
  (*R3) = (*R2) - (*R3);
  (*R2) = 2.0f * (*R2) - (*R3);

  (*R2) = (*R0) - (*R2);
  (*R0) = 2.0f * (*R0) - (*R2);
  (*R3) = (*R1) + (fvect2)((*R3).y, -(*R3).x);
  (*R1) = 2.0f * (*R1) - (*R3);

  T = (*R1);
  (*R1) = (*R2);
  (*R2) = T;

}

__attribute__((always_inline)) void
FwdPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global float2 *bufIn, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3)
{


  if(rw)
  {
    (*R0) = bufIn[inOffset + ( 0 + me * 1 + 0 + 0 ) * 1];
    (*R1) = bufIn[inOffset + ( 0 + me * 1 + 0 + 2 ) * 1];
    (*R2) = bufIn[inOffset + ( 0 + me * 1 + 0 + 4 ) * 1];
    (*R3) = bufIn[inOffset + ( 0 + me * 1 + 0 + 6 ) * 1];
  }


  FwdRad4B1(R0, R1, R2, R3);


  if(rw)
  {
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 0 ) * 1] = (*R0).x;
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 1 ) * 1] = (*R1).x;
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 2 ) * 1] = (*R2).x;
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 3 ) * 1] = (*R3).x;
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  if(rw)
  {
    (*R0).x = bufOutRe[outOffset + ( 0 + me * 2 + 0 + 0 ) * 1];
    (*R1).x = bufOutRe[outOffset + ( 0 + me * 2 + 0 + 4 ) * 1];
    (*R2).x = bufOutRe[outOffset + ( 0 + me * 2 + 1 + 0 ) * 1];
    (*R3).x = bufOutRe[outOffset + ( 0 + me * 2 + 1 + 4 ) * 1];
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  if(rw)
  {
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 0 ) * 1] = (*R0).y;
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 1 ) * 1] = (*R1).y;
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 2 ) * 1] = (*R2).y;
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 3 ) * 1] = (*R3).y;
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  if(rw)
  {
    (*R0).y = bufOutIm[outOffset + ( 0 + me * 2 + 0 + 0 ) * 1];
    (*R1).y = bufOutIm[outOffset + ( 0 + me * 2 + 0 + 4 ) * 1];
    (*R2).y = bufOutIm[outOffset + ( 0 + me * 2 + 1 + 0 ) * 1];
    (*R3).y = bufOutIm[outOffset + ( 0 + me * 2 + 1 + 4 ) * 1];
  }


  barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
FwdPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3)
{



  {
    float2 W = twiddles[3 + 1 * ((2 * me + 0) % 4) + 0];
    float TR, TI;
    TR = (W.x * (*R1).x) - (W.y * (*R1).y);
    TI = (W.y * (*R1).x) + (W.x * (*R1).y);
    (*R1).x = TR;
    (*R1).y = TI;
  }

  {
    float2 W = twiddles[3 + 1 * ((2 * me + 1) % 4) + 0];
    float TR, TI;
    TR = (W.x * (*R3).x) - (W.y * (*R3).y);
    TI = (W.y * (*R3).x) + (W.x * (*R3).y);
    (*R3).x = TR;
    (*R3).y = TI;
  }

  FwdRad2B1(R0, R1);
  FwdRad2B1(R2, R3);


  if(rw)
  {
    __global float4 *buff4g = bufOut;

    buff4g[ 1 * me + 0 + 0 ] = (float4)((*R0).x, (*R0).y, (*R2).x, (*R2).y);
    buff4g[ 1 * me + 0 + 2 ] = (float4)((*R1).x, (*R1).y, (*R3).x, (*R3).y);
  }

}

__attribute__((always_inline)) void
InvPass0(uint rw, uint b, uint me, uint inOffset, uint outOffset, __global float2 *bufIn, __local float *bufOutRe, __local float *bufOutIm, float2 *R0, float2 *R1, float2 *R2, float2 *R3)
{


  if(rw)
  {
    (*R0) = bufIn[inOffset + ( 0 + me * 1 + 0 + 0 ) * 1];
    (*R1) = bufIn[inOffset + ( 0 + me * 1 + 0 + 2 ) * 1];
    (*R2) = bufIn[inOffset + ( 0 + me * 1 + 0 + 4 ) * 1];
    (*R3) = bufIn[inOffset + ( 0 + me * 1 + 0 + 6 ) * 1];
  }


  InvRad4B1(R0, R1, R2, R3);


  if(rw)
  {
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 0 ) * 1] = (*R0).x;
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 1 ) * 1] = (*R1).x;
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 2 ) * 1] = (*R2).x;
    bufOutRe[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 3 ) * 1] = (*R3).x;
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  if(rw)
  {
    (*R0).x = bufOutRe[outOffset + ( 0 + me * 2 + 0 + 0 ) * 1];
    (*R1).x = bufOutRe[outOffset + ( 0 + me * 2 + 0 + 4 ) * 1];
    (*R2).x = bufOutRe[outOffset + ( 0 + me * 2 + 1 + 0 ) * 1];
    (*R3).x = bufOutRe[outOffset + ( 0 + me * 2 + 1 + 4 ) * 1];
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  if(rw)
  {
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 0 ) * 1] = (*R0).y;
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 1 ) * 1] = (*R1).y;
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 2 ) * 1] = (*R2).y;
    bufOutIm[outOffset + ( ((1 * me + 0) / 1) * 4 + (1 * me + 0) % 1 + 3 ) * 1] = (*R3).y;
  }


  barrier(CLK_LOCAL_MEM_FENCE);

  if(rw)
  {
    (*R0).y = bufOutIm[outOffset + ( 0 + me * 2 + 0 + 0 ) * 1];
    (*R1).y = bufOutIm[outOffset + ( 0 + me * 2 + 0 + 4 ) * 1];
    (*R2).y = bufOutIm[outOffset + ( 0 + me * 2 + 1 + 0 ) * 1];
    (*R3).y = bufOutIm[outOffset + ( 0 + me * 2 + 1 + 4 ) * 1];
  }


  barrier(CLK_LOCAL_MEM_FENCE);

}

__attribute__((always_inline)) void
InvPass1(uint rw, uint b, uint me, uint inOffset, uint outOffset, __local float *bufInRe, __local float *bufInIm, __global float2 *bufOut, float2 *R0, float2 *R1, float2 *R2, float2 *R3)
{



  {
    float2 W = twiddles[3 + 1 * ((2 * me + 0) % 4) + 0];
    float TR, TI;
    TR = (W.x * (*R1).x) + (W.y * (*R1).y);
    TI = -(W.y * (*R1).x) + (W.x * (*R1).y);
    (*R1).x = TR;
    (*R1).y = TI;
  }

  {
    float2 W = twiddles[3 + 1 * ((2 * me + 1) % 4) + 0];
    float TR, TI;
    TR = (W.x * (*R3).x) + (W.y * (*R3).y);
    TI = -(W.y * (*R3).x) + (W.x * (*R3).y);
    (*R3).x = TR;
    (*R3).y = TI;
  }

  InvRad2B1(R0, R1);
  InvRad2B1(R2, R3);


  if(rw)
  {
    __global float4 *buff4g = bufOut;

    buff4g[ 1 * me + 0 + 0 ] = (float4)((*R0).x, (*R0).y, (*R2).x, (*R2).y) * 1.2500000000000000e-01f;
    buff4g[ 1 * me + 0 + 2 ] = (float4)((*R1).x, (*R1).y, (*R3).x, (*R3).y) * 1.2500000000000000e-01f;
  }

}

typedef union  { uint u;
                 int i;
} cb_t;

__kernel __attribute__((reqd_work_group_size (64,1,1)))
//void fft_fwd(__constant cb_t *cb __attribute__((max_constant_size(32))), __global const float2 * restrict gbIn, __global float2 * restrict gbOut)
void fft_fwd(__global const float2 * restrict gbIn, __global float2 * restrict gbOut)
{
  uint me = get_local_id(0);
  uint batch = get_group_id(0);

  __local float lds[256];

  uint iOffset;
  uint oOffset;
  __global float2 *lwbIn;
  __global float2 *lwbOut;

  float2 R0, R1, R2, R3;

  //uint rw = (me < ((cb[0].u) - batch*32)*2) ? 1 : 0;
  uint rw = (me < ((1) - batch * 32) * 2) ? 1 : 0;

  uint b = 0;

  iOffset = (batch * 32 + (me / 2)) * 8;
  oOffset = (batch * 32 + (me / 2)) * 8;
  lwbIn = gbIn + iOffset;
  lwbOut = gbOut + oOffset;

  FwdPass0(rw, b, me % 2, 0, (me / 2) * 8, lwbIn, lds, lds, &R0, &R1, &R2, &R3);
  FwdPass1(rw, b, me % 2, (me / 2) * 8, 0, lds, lds, lwbOut, &R0, &R1, &R2, &R3);
}

__kernel __attribute__((reqd_work_group_size (64,1,1)))
//void fft_back(__constant cb_t *cb __attribute__((max_constant_size(32))), __global const float2 * restrict gbIn, __global float2 * restrict gbOut)
void fft_back(__global const float2 * restrict gbIn, __global float2 * restrict gbOut)
{
  uint me = get_local_id(0);
  uint batch = get_group_id(0);

  __local float lds[256];

  uint iOffset;
  uint oOffset;
  __global float2 *lwbIn;
  __global float2 *lwbOut;

  float2 R0, R1, R2, R3;

  //uint rw = (me < ((cb[0].u) - batch*32)*2) ? 1 : 0;
  uint rw = (me < ((1) - batch * 32) * 2) ? 1 : 0;

  uint b = 0;

  iOffset = (batch * 32 + (me / 2)) * 8;
  oOffset = (batch * 32 + (me / 2)) * 8;
  lwbIn = gbIn + iOffset;
  lwbOut = gbOut + oOffset;

  InvPass0(rw, b, me % 2, 0, (me / 2) * 8, lwbIn, lds, lds, &R0, &R1, &R2, &R3);
  InvPass1(rw, b, me % 2, (me / 2) * 8, 0, lds, lds, lwbOut, &R0, &R1, &R2, &R3);
}



#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// DCT Constants
__device__ constexpr float a = 1.387039845322148f;    // sqrt(2) * cos(    pi / 16)
__device__ constexpr float b = 1.306562964876377f;    // sqrt(2) * cos(    pi /  8)
__device__ constexpr float c = 1.175875602419359f;    // sqrt(2) * cos(3 * pi / 16)
__device__ constexpr float d = 0.785694958387102f;    // sqrt(2) * cos(5 * pi / 16)
__device__ constexpr float e = 0.541196100146197f;    // sqrt(2) * cos(3 * pi /  8)
__device__ constexpr float f = 0.275899379282943f;    // sqrt(2) * cos(7 * pi / 16)
__device__ constexpr float norm_factor = 0.3535533905932737f;  // 1 / sqrt(8)

// DCT forward 1D implementation
__device__ void dct_fwd_8x8_1d(float *vecf8) 
{
    float x0 = vecf8[0];
    float x1 = vecf8[1];
    float x2 = vecf8[2];
    float x3 = vecf8[3];
    float x4 = vecf8[4];
    float x5 = vecf8[5];
    float x6 = vecf8[6];
    float x7 = vecf8[7];

    float tmp0 = x0 + x7;
    float tmp1 = x1 + x6;
    float tmp2 = x2 + x5;
    float tmp3 = x3 + x4;
    float tmp4 = x0 - x7;
    float tmp5 = x6 - x1;
    float tmp6 = x2 - x5;
    float tmp7 = x4 - x3;

    float tmp8 = tmp0 + tmp3;
    float tmp9 = tmp0 - tmp3;
    float tmp10 = tmp1 + tmp2;
    float tmp11 = tmp1 - tmp2;

    x0 = norm_factor * (tmp8 + tmp10);
    x2 = norm_factor * (b * tmp9 + e * tmp11);
    x4 = norm_factor * (tmp8 - tmp10);
    x6 = norm_factor * (e * tmp9 - b * tmp11);

    x1 = norm_factor * (a * tmp4 - c * tmp5 + d * tmp6 - f * tmp7);
    x3 = norm_factor * (c * tmp4 + f * tmp5 - a * tmp6 + d * tmp7);
    x5 = norm_factor * (d * tmp4 + a * tmp5 + f * tmp6 - c * tmp7);
    x7 = norm_factor * (f * tmp4 + d * tmp5 + c * tmp6 + a * tmp7);

    vecf8[0] = x0;
    vecf8[1] = x1;
    vecf8[2] = x2;
    vecf8[3] = x3;
    vecf8[4] = x4;
    vecf8[5] = x5;
    vecf8[6] = x6;
    vecf8[7] = x7;
}

__device__ void YCbCr_hip_compute(float *srcPtr,d_float8 *Ch1_f8, d_float8 *Ch2_f8,d_float8 *Ch3_f8)
{
    d_float8 Y_f8,Cb_f8,Cr_f8;

    Y_f8->f4[0]  = Ch1_f8->f4[0] * (float4)0.299 + Ch2_f8->f4[0] * (float4)0.587 + Ch3_f8->f4[0] * float4(0.114) ;
    Y_f8->f4[1]  = Ch1_f8->f4[1] * (float4)0.299 + Ch2_f8->f4[1] * (float4)0.587 + Ch3_f8->f4[1] * float4(0.114) ;

    Cb_f8->f4[0] = Ch1_f8->f4[0] * (float4)(-0.168736) + Ch2_f8->f4[0] * (float4)(-0.331264) + Ch3_f8->f4[0] * (float4)0.5 + (float4)128;
    Cb_f8->f4[1] = Ch1_f8->f4[1] * (float4)(-0.168736) + Ch2_f8->f4[1] * (float4)(-0.331264) + Ch3_f8->f4[1] * (float4)0.5 + (float4)128;

    Cr_f8->f4[0] = Ch1_f8->f4[0] * (float4)0.5 + Ch2_f8->f4[0] * (float4)(-0.418688) + Ch3_f8->f4[0] * (float4)(-0.081312) + (float4)128;
    Cr_f8->f4[1] = Ch1_f8->f4[1] * (float4)0.5 + Ch2_f8->f4[1] * (float4)(-0.418688) + Ch3_f8->f4[1] * (float4)(-0.081312) + (float4)128;

    //Storing the results back into the shared memory (Inplace)
    *Ch1_f8 =  Y_f8; 
    *Ch2_f8 =  Cb_f8;  
    *Ch3_f8 =  Cr_f8;  

}


__device__ void verticalDownSampling(float *srcPtr,d_float8 *Cb_f8_1, d_float8 *Cb_f8_2,d_float8 *Cr_f8_1, d_float8 *Cr_f8_2)
{
    //Storing the results back into the shared memory (Inplace)
	Cb_f8_1->f4[0] = (Cb_f8_1->f4[0] + Cb_f8_2->f4[0]) * 0.5f;
	Cb_f8_1->f4[1] = (Cb_f8_1->f4[1] + Cb_f8_2->f4[1]) * 0.5f;
    Cr_f8_1->f4[0] = (Cr_f8_1->f4[0] + Cr_f8_2->f4[0]) * 0.5f;
	Cr_f8_1->f4[1] = (Cr_f8_1->f4[1] + Cr_f8_2->f4[1]) * 0.5f;
}

__device__ void horizontalDownSampling(float *srcPtr,d_float8 *Cb_f8_1, d_float8 *Cb_f8_2,d_float8 *Cr_f8_1, d_float8 *Cr_f8_2,d_float8 *Cb, d_float8 *Cr)
{
    //Each thread carries 8 elements (float8) per channel add odd elements to even elements and * 0.5
    d_float8 odds,evens;
    evens->f4[0] = make_float4(Cb_f8_1->f4[0].x,Cb_f8_1->f4[0].z,Cb_f8_1->f4[1].x,Cb_f8_1->f4[1].z) ;
    evens->f4[1] = make_float4(Cb_f8_2->f4[0].x,Cb_f8_2->f4[0].z,Cb_f8_2->f4[1].x,Cb_f8_2->f4[1].z) ;
    odds->f4[0] = make_float4(Cb_f8_1->f4[0].y,Cb_f8_1->f4[0].w,Cb_f8_1->f4[1].y,Cb_f8_1->f4[1].w) ;
    odds->f4[1] = make_float4(Cb_f8_2->f4[0].y,Cb_f8_2->f4[0].w,Cb_f8_2->f4[1].y,Cb_f8_2->f4[1].w) ;

    // Horizontal average for Cb and Store the results back in the first d_float8 in  Cb
    *Cb = (evens + odds) * 0.5f;

    // Repeat the process for Cr
    evens->f4[0] = make_float4(Cr_f8_1->f4[0].x,Cr_f8_1->f4[0].z,Cr_f8_1->f4[1].x,Cr_f8_1->f4[1].z) ;
    evens->f4[1] = make_float4(Cr_f8_2->f4[0].x,Cr_f8_2->f4[0].z,Cr_f8_2->f4[1].x,Cr_f8_2->f4[1].z) ;
    odds->f4[0] = make_float4(Cr_f8_1->f4[0].y,Cr_f8_1->f4[0].w,Cr_f8_1->f4[1].y,Cr_f8_1->f4[1].w) ;
    odds->f4[1] = make_float4(Cr_f8_2->f4[0].y,Cr_f8_2->f4[0].w,Cr_f8_2->f4[1].y,Cr_f8_2->f4[1].w) ;

    // Horizontal average for Cr and Store the results back in the first d_float8 in  Cr
    *Cr = (evens + odds) * 0.5f;
}

template <typename T>
__global__ void jpeg_compression_distortion_pkd3_hip_tensor( T *srcPtr,
                                                                uint2 srcStridesNH,
                                                                T *dstPtr,
                                                                uint3 dstStridesNCH,
                                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int hipThreadIdx_x16 = hipThreadIdx_x8 * 2;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint downsampled_Idx = (id_z * dstStridesNCH.x) + (id_y / 2 * dstStridesNCH.z) + (id_x / 2);
    
    d_float24 src_f24, dst_f24;
    // 16 Rows x 3 Channels and 16 x 8 columns with each element being a float8 
    __shared__ float src_smem[16*3][16*8];

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];
        
    if ((id_y < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (id_x < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        *(uint2 *)src_smem_channel[0] = (uint2)0;
        *(uint2 *)src_smem_channel[1] = (uint2)0;
        *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    //RGB to YCbCr
    YCbCr_hip_compute(srcPtr,&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    //Downsampling
    int CbCry = hipThreadIdx_y * 2;
    if(CbCry < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
    {
        verticalDownSampling(srcPtr,
                             &src_smem[16 + CbCry][hipThreadIdx_x8],
                             &src_smem[16 + CbCry + 1][hipThreadIdx_x8],
                             &src_smem[32 + CbCry][hipThreadIdx_x8],
                             &src_smem[32 + CbCry + 1][hipThreadIdx_x8]);
        __syncthreads();
        d_float8 Cb,Cr;
        horizontalDownSampling(srcPtr,
                               &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x16],
                               &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x16 + 8],
                               &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x16],
                               &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x16 + 8],
                               &Cb,
                               &Cr);

        src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = Cb;
        //src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = Cr;
        //Storing Cr beside Cb block
        src_smem[hipThreadIdx_y_channel.y][64 + hipThreadIdx_x8] = Cr;
        __syncthreads();
    }
    //1D row wise DCT for Y channel
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    //1D row wise DCT for Cb and Cr channels but should only done for 8 rows
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);  
    //Now for each column we should do DCT

    //So by now in smem first [0 to 16] x (16 x 8) has Y, [16 to 24] x (8 x 8) has Cb and [16 to 24] x (8 x 8) has Cr
    //We have 16 x 16 threads on X and Y dimension and 128 elements in each row
    


    //Divide into 8x8 blocks 

    //Perform DCT on each layer of Y Cb Cr

    //Quantization on each layer of Y Cb Cr

    //rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
RppStatus hip_exec_jpeg_compression_distortion( T *srcPtr,
                                                RpptGenericDescPtr srcGenericDescPtr,
                                                T *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Rpp32u *roiTensor,
                                                rpp::Handle& handle)
{
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(jpeg_compression_distortion_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr);

    return RPP_SUCCESS;
}
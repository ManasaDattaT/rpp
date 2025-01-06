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
__device__ void dct_fwd_8x8_1d(float *vecf8, int stride,int col, bool row) 
{
    if(row)
    {
        float4 x_idx1,x_idx2;
        x_idx1 = make_float4(0,1,2,3)
        x_idx2 = make_float4(4,5,6,7)
    }
    else
    {
       x_idx1 = (float4)col;
       x_idx2 = (float4)col;
    }

    //Adjust for rows and columns
    float x0 = vecf8[0 * stride + x_idx1.x];
    float x1 = vecf8[1 * stride + x_idx1.y];
    float x2 = vecf8[2 * stride + x_idx1.z];
    float x3 = vecf8[3 * stride + x_idx1.w];
    float x4 = vecf8[4 * stride + x_idx2.x];
    float x5 = vecf8[5 * stride + x_idx2.y];
    float x6 = vecf8[6 * stride + x_idx2.z];
    float x7 = vecf8[7 * stride + x_idx2.w];

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

    vecf8[0 * stride + x_idx1.x] = x0;
    vecf8[1 * stride + x_idx1.y] = x1;
    vecf8[2 * stride + x_idx1.z] = x2;
    vecf8[3 * stride + x_idx1.w] = x3;
    vecf8[4 * stride + x_idx2.x] = x4;
    vecf8[5 * stride + x_idx2.y] = x5;
    vecf8[6 * stride + x_idx2.z] = x6;
    vecf8[7 * stride + x_idx2.w] = x7;
}

__device__ void dct_inv_8x8_1d(float *vecf8, int stride,int col, bool row) 
{
    if(row)
    {
        float4 x_idx1,x_idx2;
        x_idx1 = make_float4(0,1,2,3)
        x_idx2 = make_float4(4,5,6,7)
    }
    else
    {
       x_idx1 = (float4)col;
       x_idx2 = (float4)col;
    }

    //Adjust for rows and columns
    float x0 = vecf8[0 * stride + x_idx1.x];
    float x1 = vecf8[1 * stride + x_idx1.y];
    float x2 = vecf8[2 * stride + x_idx1.z];
    float x3 = vecf8[3 * stride + x_idx1.w];
    float x4 = vecf8[4 * stride + x_idx2.x];
    float x5 = vecf8[5 * stride + x_idx2.y];
    float x6 = vecf8[6 * stride + x_idx2.z];
    float x7 = vecf8[7 * stride + x_idx2.w];

    float tmp0 = x0 + x4;
    float tmp1 = b * x2 + e * x6;

    float tmp2 = tmp0 + tmp1;
    float tmp3 = tmp0 - tmp1;
    float tmp4 = f * x7 + a * x1 + c * x3 + d * x5;
    float tmp5 = a * x7 - f * x1 + d * x3 - c * x5;

    float tmp6 = x0 - x4;
    float tmp7 = e * x2 - b * x6;

    float tmp8 = tmp6 + tmp7;
    float tmp9 = tmp6 - tmp7;
    float tmp10 = c * x1 - d * x7 - f * x3 - a * x5;
    float tmp11 = d * x1 + c * x7 - a * x3 + f * x5;

    x0 = norm_factor * (tmp2 + tmp4);
    x7 = norm_factor * (tmp2 - tmp4);
    x4 = norm_factor * (tmp3 + tmp5);
    x3 = norm_factor * (tmp3 - tmp5);

    x1 = norm_factor * (tmp8 + tmp10);
    x5 = norm_factor * (tmp9 - tmp11);
    x2 = norm_factor * (tmp9 + tmp11);
    x6 = norm_factor * (tmp8 - tmp10);
  
    vecf8[0 * stride + x_idx1.x] = x0;
    vecf8[1 * stride + x_idx1.y] = x1;
    vecf8[2 * stride + x_idx1.z] = x2;
    vecf8[3 * stride + x_idx1.w] = x3;
    vecf8[4 * stride + x_idx2.x] = x4;
    vecf8[5 * stride + x_idx2.y] = x5;
    vecf8[6 * stride + x_idx2.z] = x6;
    vecf8[7 * stride + x_idx2.w] = x7;
}

__device__ void YCbCr_hip_compute(d_float8 *Ch1_f8, d_float8 *Ch2_f8,d_float8 *Ch3_f8)
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

__device__ void verticalDownSampling(d_float8 *Cb_f8_1, d_float8 *Cb_f8_2,d_float8 *Cr_f8_1, d_float8 *Cr_f8_2)
{
    //Storing the results back into the shared memory (Inplace)
	Cb_f8_1->f4[0] = (Cb_f8_1->f4[0] + Cb_f8_2->f4[0]) * 0.5f;
	Cb_f8_1->f4[1] = (Cb_f8_1->f4[1] + Cb_f8_2->f4[1]) * 0.5f;
    Cr_f8_1->f4[0] = (Cr_f8_1->f4[0] + Cr_f8_2->f4[0]) * 0.5f;
	Cr_f8_1->f4[1] = (Cr_f8_1->f4[1] + Cr_f8_2->f4[1]) * 0.5f;
}

__device__ void horizontalDownSampling(d_float8 *Cb_f8_1, d_float8 *Cb_f8_2,d_float8 *Cr_f8_1, d_float8 *Cr_f8_2,d_float8 *Cb, d_float8 *Cr)
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

__device__ __inline__ float quantize(float* value, float* coeff) {
    for(int i=0; i<8 ;i++)
        coeff[i] * roundf(value[i] * __frcp_rn(coeff[i]));
}

template <typename T>
__global__ void jpeg_compression_distortion_pkd3_hip_tensor( T *srcPtr,
                                                             uint2 srcStridesNH,
                                                             T *dstPtr,
                                                             uint3 dstStridesNCH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             float *Ytable,
                                                             float *CBCrtable,
                                                             float q_scale)
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
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,0,true);
    //1D row wise DCT for Cb and Cr channels but should only done for 8 rows but here being done for 16 rows
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,0,true);  

    // After row-wise DCT and synchronization
    __syncthreads();
    //Now for each column we should do DCT

    //So by now in smem first [0 to 16] x (16 x 8) has Y, [16 to 24] x (8 x 8) has Cb and [16 to 24] x (8 x 8) has Cr
    //We have 16 x 16 threads on X and Y dimension and 128 elements in each row
    // Each thread takes one column (total 128 threads for 128 columns)
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128) { 
        // 1D DCT for First 8 rows (Y channel)   
        dct_fwd_8x8_1d(&src_smem[0][col], 128,col,false);
      
        // 1D DCT for Second 8 rows (Y channel)
        dct_fwd_8x8_1d(&src_smem[8][col], 128,col,false);
        
        // 1D DCT for 8 rows (Cb/Cr channels)
        dct_fwd_8x8_1d(&src_smem[16][col],128,col,false);
    }
    __syncthreads();
    //Quantization on each layer of Y Cb Cr
    //For every 8 elements row wise, we are taking respected row in the table and doing quantization
    //For Y
    quantize(src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);
    //For Cb Cr
    //Here we can do this for only 8 threads in Y, this and in row wise DCT
    quantize(src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);  
    __syncthreads();


    //rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
RppStatus hip_exec_jpeg_compression_distortion( T *srcPtr,
                                                RpptGenericDescPtr srcGenericDescPtr,
                                                T *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Rpp32u *roiTensor,
                                                rpp::Handle& handle,
                                                int quality)
{
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int *h_Ytable, *h_CBCrtable;
    hipHostMalloc(&h_Ytable, sizeof(int) * 64, hipHostMallocMapped);
    hipHostMalloc(&h_CBCrtable, sizeof(int) * 64, hipHostMallocMapped);

    quality = clamp<int>(quality, 1, 100);
    float q_scale = 1.0f;
    if (quality < 50) {
        q_scale = 50.0f / quality;
    } else {
        q_scale = 2.0f - (2 * quality / 100.0f);
    }

    // Initialize the Ytable
    int Ytable[8][8] = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };

    // Initialize the CBCrtable
    int CBCrtable[8][8] = {
        {17, 18, 24, 47, 99, 99, 99, 99},
        {18, 21, 26, 66, 99, 99, 99, 99},
        {24, 26, 56, 99, 99, 99, 99, 99},
        {47, 66, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99}
    };
    for (int i = 0; i < 8; i++) 
    {
        for (int j = 0; j < 8; j++) 
        {
            Ytable[i][j] = std::max<uint8_t>(ConvertSat<uint8_t>(q_scale * table[i][j]), 1);
            CBCrtable[i][j] = std::max<uint8_t>(ConvertSat<uint8_t>(q_scale * table[i][j]), 1);
        }
    }

    // Copy the tables to pinned memory
    memcpy(h_Ytable, Ytable, sizeof(int) * 64);
    memcpy(h_CBCrtable, CBCrtable, sizeof(int) * 64);

    // Get the device pointers for pinned memory
    int *d_Ytable, *d_CBCrtable;
    hipHostGetDevicePointer(&d_Ytable, h_Ytable, 0);
    hipHostGetDevicePointer(&d_CBCrtable, h_CBCrtable, 0);

    hipLaunchKernelGGL(jpeg_compression_distortion_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       d_Ytable,
                       d_CBCrtable,
                       q_scale);

    return RPP_SUCCESS;
}
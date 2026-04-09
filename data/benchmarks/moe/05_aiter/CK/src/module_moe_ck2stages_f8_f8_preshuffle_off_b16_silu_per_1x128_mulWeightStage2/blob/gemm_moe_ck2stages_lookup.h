#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

#define GENERATE_LOOKUP_TABLE()                                                                                      \
   {                                                                                                                             \
       {"moe_ck2stages_gemm1_256x16x128x256_1x4_MulABScaleExpertWeightA8W8blkscale_v1_Nswizzle0_Quant4_MulRoutedWeight0_silu_F8_F8_B16",                                                                                                       \
        ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256, 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>},                       \
       {"moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleExpertWeightA8W8blkscale_v3_Nswizzle0_Quant4_MulRoutedWeight0_silu_F8_F8_B16",                                                                                                       \
        ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128, 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>},                       \
       {"moe_ck2stages_gemm2_256x16x128x256_1x4_MulABScaleExpertWeightA8W8blkscale_v1_Nswizzle0_Quant4_MulRoutedWeight1_F8_F8_B16",                                                                                                       \
        ck_moe_stage2_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256, 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>},                       \
       {"moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightA8W8blkscale_v3_Nswizzle0_Quant4_MulRoutedWeight1_F8_F8_B16",                                                                                                       \
        ck_moe_stage2_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128, 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>},                       \
   }


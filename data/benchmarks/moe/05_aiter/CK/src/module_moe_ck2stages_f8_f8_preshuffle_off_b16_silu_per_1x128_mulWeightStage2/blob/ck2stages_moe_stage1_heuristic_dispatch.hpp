#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage1_heuristic_dispatch(int block_m, at::ScalarType x_dtype, at::ScalarType w_dtype, at::ScalarType y_dtype, int act_op, int quant, bool mul_routed_weight_stage)
{{

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    TORCH_CHECK(
        false,
        "Unsupported kernel config for moe heuristic dispatch");
}}


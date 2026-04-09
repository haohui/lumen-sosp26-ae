// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common_blockscale.cuh"

using A0DataType = F8;
using B0DataType = F8;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = MulABScaleExpertWeightA8W8blkscale;
const bool Nswizzle = false;
const bool PerTensorQuant = 4 == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = true;
const int ActOP = 1;
CK_MOE_STAGE1_GEMM_DEFINE(256, 16, 128, 256, 1, 4, V1)

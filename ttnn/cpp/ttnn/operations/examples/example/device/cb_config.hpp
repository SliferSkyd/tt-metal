#pragma once

#include "hostdevcommon/kernel_structs.h"

namespace cb_backward {
inline constexpr auto output_grad = tt::CBIndex::c_0;
inline constexpr auto input = tt::CBIndex::c_1;
inline constexpr auto output = tt::CBIndex::c_2;
inline constexpr auto mask_w = tt::CBIndex::c_3;
inline constexpr auto input_grad = tt::CBIndex::c_4;
}  // namespace cb_backward

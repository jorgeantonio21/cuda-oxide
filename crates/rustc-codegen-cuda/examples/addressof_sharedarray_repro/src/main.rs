/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Source-level repro for `llvm.addressof` export ordering.
//!
//! Build-only regression:
//!   cargo oxide build addressof_sharedarray_repro --emit-nvvm-ir --arch sm_90

#![allow(static_mut_refs)]
#![allow(clippy::assign_op_pattern)] // Expanded assignment preserves the addressof repro CFG.

use cuda_device::{SharedArray, cuda_module, device, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn sharedarray_late_use() {
        static mut OUTPUT_NORM: SharedArray<f32, 1> = SharedArray::UNINIT;

        unsafe {
            let weight = repro_weight();
            OUTPUT_NORM[0] = OUTPUT_NORM[0] * weight;
        }
    }

    #[inline(never)]
    #[device]
    fn repro_weight() -> f32 {
        1.0
    }
}

fn main() {
    println!("SUCCESS: addressof_sharedarray_repro build-only fixture");
}

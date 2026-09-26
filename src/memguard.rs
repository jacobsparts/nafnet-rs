//! What a pass will need, and what there is - for the CPU backend, whose
//! memory is the host's and is therefore nobody's to overspend.
//!
//! THE GPU SIDE HAS THE SAME QUESTION AND ANSWERS IT IN `gpu::PlanShape`, where
//! the answer is EXACT because every buffer is one the engine allocates itself
//! and holds for the pass. The CPU side cannot be exact: `net::forward_cpu` works
//! in `Vec<f32>`s whose growth the allocator decides, and it holds them for as
//! long as the borrow checker lets it. So this is a MODEL, and the model is
//! deliberately conservative in one direction - it is better to refuse a pass
//! that would have fitted than to start one that swaps the machine.
//!
//! THE MODEL IS THE LIVE SET, which is what the CPU pass actually holds:
//!
//!   * one activation per encoder level, because the decoder reads each one back
//!     as a skip - that is the term that scales with the image;
//!   * the activation in flight and the bottleneck;
//!   * the input and output planes, which are both full size;
//!   * and the block scratch, at its widest, which is SEVEN planes of the
//!     level-0 shape (see the slot table in `net::Scratch`) plus the decoder's
//!     pixel-shuffle temporary, and which is allocated once and never shrinks.
//!
//! MEASURED AGAINST `Maximum resident set size` on six image sizes, this lands
//! 1.20x to 1.38x UNDER the real peak - the gap is allocator growth slack and the
//! mapped weights - so `CpuPlan::peak` applies a factor of 1.25 on top and the
//! guard below compares that. See the README's memory table for the readings.

use crate::weights::Config;

/// The CPU pass's memory, as arithmetic, before it allocates any of it.
pub struct CpuPlan {
    /// The activations: every skip, the bottleneck, the input and the output.
    pub live: usize,
    /// The block scratch at its widest, which is one allocation for the pass.
    pub scratch: usize,
    /// `live + scratch`, times the measured slack factor. This is the number the
    /// guard compares against what the machine has.
    pub peak: usize,
}

/// The factor between the modelled live set and the measured peak RSS. Measured
/// across six sizes (1.20x at 2048x2048 up to 1.38x at 640x480); 1.25 is the
/// round number above the large-image end, which is where a refusal matters.
pub const SLACK: f64 = 1.25;

/// The fixed part of the footprint: the checkpoint's tensors, the process, and
/// the allocator's own bookkeeping. It is added AFTER the slack because it is not
/// proportional to the image - at 640x480 the model alone under-predicts the
/// measured peak, and this is what closes that gap. Measured as 45 MiB at
/// 640x480; 96 is the same round number with room to spare.
pub const BASE: usize = 96 * 1024 * 1024;

impl CpuPlan {
    /// The plan for `h`x`wd` - the PADDED geometry, which is what the pass runs.
    pub fn of(cfg: &Config, h: usize, wd: usize) -> CpuPlan {
        let levels = cfg.levels();
        let mut live = 0usize;
        let mut scratch = 0usize;
        for l in 0..levels {
            let c = cfg.width_at(l);
            let (hh, ww) = ((h >> l).max(1), (wd >> l).max(1));
            let hw = hh * ww;
            live += c * hw * 4;
            // S_A c*hw, S_B 2c*hw, S_C 2c*hw, S_D c*hw, S_E c*hw - seven planes
            // of the level's own width, and the widest level wins.
            scratch = scratch.max(7 * c * hw * 4);
        }
        let (bh, bw) = ((h >> levels).max(1), (wd >> levels).max(1));
        // The bottleneck, at `width << levels` channels on the smallest plane.
        live += (cfg.width << levels) * bh * bw * 4;
        // The decoder's pixel-shuffle temporary: `2 * c_before` channels of the
        // mirrored level's plane, where `c_before` is the level ABOVE it.
        for ml in 0..levels {
            let c_before = cfg.width << (ml + 1);
            let (hh, ww) = ((h >> ml).max(1), (wd >> ml).max(1));
            scratch = scratch.max(2 * c_before * hh * ww * 4);
        }
        // The input plane and the output plane, both full size.
        live += 3 * h * wd * 4 * 2;
        let peak = ((live + scratch) as f64 * SLACK) as usize + BASE;
        CpuPlan { live, scratch, peak }
    }
}

/// What the kernel says is available for a new allocation, in bytes.
///
/// `MemAvailable` AND NOT `MemFree`: free memory is the part that is already
/// unused, while available is what a process can actually get without swapping -
/// it counts reclaimable page cache, which on a machine that has read a 1.2 GB
/// checkpoint is most of it. Guarding on `MemFree` would refuse passes that fit.
///
/// `None` when the file is unreadable or has no such line, and the caller then
/// does not guard at all: refusing every pass because a `/proc` line moved would
/// be worse than running one that might swap.
pub fn host_available() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        // A `?` HERE WOULD RETURN FROM THE WHOLE FUNCTION ON THE FIRST LINE THAT
        // DOES NOT MATCH, and `/proc/meminfo`'s first line is `MemTotal:` - so
        // this loop skipped straight to the `None` at the bottom and the guard
        // silently did nothing. A missing key is a `continue`, not a failure.
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let kb: usize = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
        return Some(kb * 1024);
    }
    None
}

//! The CUDA half of the engine: the two fatbins, the kernel-name sets, and the
//! per-launch profile hook.
//!
//! WHICH MODULE A KERNEL LIVES IN IS A PROPERTY OF THE NAME, and the two
//! modules are disjoint: `cuda/kernels.cu` (toolkit) and `cuda/nafnet.cu` (this
//! engine). `resolve` asks the project module first and then the toolkit, so a
//! kernel promoted from one to the other is a one-line change here rather than
//! a debugging session. A name in NEITHER module fails at startup, where the
//! message can say which of the two lists is wrong.
use lightgpu::vm::{Module, DevBuf, Launch};

/// The toolkit's kernels, compiled from lightgpu's `cuda/kernels.cu`.
///
/// These are the ones the graph launches: every NAFBlock is built from
/// `lg_conv1x1`, `lg_conv3x3s1p1`, `lg_channel_layer_norm`, `lg_channel_mean`,
/// `lg_mul` and `lg_channel_scale`, and the graph adds `lg_add`.
///
/// `lg_conv1x1` IS RESOLVED BUT NO LONGER LAUNCHED - `nf_conv1x1_oc` replaced it
/// (see `Gpu::conv1x1`) - and it is kept here only because `--cuda-selftest`
/// still compares the replacement against it on identical inputs. A name this
/// list does not carry cannot be launched at all, which is what makes the list
/// worth reading: everything here is either in the graph or in the selftest.
///
/// `lg_add_scaled` used to be here and is gone: its `s` is a WHOLE-PLANE scalar
/// and every scale the graph applies is per CHANNEL, so `nf_residual` covers the
/// only case it was resolved for.
pub const TOOLKIT_KERNELS: &[&str] = &[
    "lg_conv3x3s1p1",
    "lg_conv1x1",
    "lg_channel_layer_norm",
    "lg_channel_mean",
    "lg_mul",
    "lg_channel_scale",
    "lg_add",
];

/// This project's own kernels, from `cuda/nafnet.cu`.
pub const PROJECT_KERNELS: &[&str] =
    &["nf_conv3x3_dw", "nf_pixel_shuffle2", "nf_down2x2s2", "nf_conv1x1_oc", "nf_residual"];

const TOOLKIT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nafnet_toolkit.fatbin"));
const PROJECT_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nafnet_project.fatbin"));

/// Driver + module state. Constructing it loads both fatbins and resolves every
/// kernel the executor can launch, so a name that is in neither module fails
/// here rather than at the first forward pass.
pub struct Cuda {
    toolkit: Module,
    project: Module,
}

impl Cuda {
    pub fn new() -> Result<Cuda, String> {
        let toolkit = Module::load(TOOLKIT_FATBIN)?;
        let project = Module::load(PROJECT_FATBIN)?;
        for k in PROJECT_KERNELS {
            if !project.has(k) {
                return Err(format!(
                    "kernel `{k}` is not in the project fatbin - is it listed in build.rs's \
                     PROJECT_KERNELS and defined in cuda/nafnet.cu?"
                ));
            }
        }
        for k in TOOLKIT_KERNELS {
            if !toolkit.has(k) {
                return Err(format!(
                    "kernel `{k}` is not in the toolkit fatbin - is it in lightgpu's \
                     src/ops/mod.rs NAMES and cuda/kernels.cu?"
                ));
            }
        }
        Ok(Cuda { toolkit, project })
    }

    /// The module a kernel lives in. Project first, then toolkit.
    pub fn module_of(&self, name: &str) -> &Module {
        if self.project.has(name) {
            &self.project
        } else {
            &self.toolkit
        }
    }

    /// Launch `name` with the given arguments.
    pub fn run(&self, name: &str, launch: Launch, args: &mut lightgpu::vm::Args) -> Result<(), String> {
        args.launch(self.module_of(name), name, launch)
    }

    /// Allocate a device buffer of `n` elements.
    pub fn buf(&self, n: usize) -> Result<DevBuf, String> {
        DevBuf::zeros(n * std::mem::size_of::<f32>())
    }

    pub fn upload(&self, v: &[f32]) -> Result<DevBuf, String> {
        DevBuf::from_host(v)
    }
}

/// Grid for `n` elements at `block` threads.
pub fn grid_for(n: usize, block: usize) -> (u32, u32, u32) {
    (((n + block - 1) / block) as u32, 1, 1)
}

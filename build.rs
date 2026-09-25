//! Compiles this engine's kernels into two modules: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this project's own
//! `cuda/nafnet.cu` (PROJECT_KERNELS). Each compiles to its own fatbin with its
//! own `--entries` list and `src/cuda.rs` loads them as separate modules, so
//! neither can shadow a name in the other.
//!
//! A kernel missing from its list is PRUNED from the fatbin and fails at launch
//! rather than at build time, so both lists are checked against the source they
//! are compiled from before nvcc runs: a typo, or a kernel moved between files,
//! fails the build.

/// Generic ops from the shared toolkit. Exactly the ones `exec_gpu` launches:
/// an entry here is a kernel embedded in the binary, so a name that no op calls
/// is wasted bytes.
const TOOLKIT_KERNELS: &[&str] = &[
    // 3x3 pad-1 conv, the encoder/decoder input convs and `intro`/`ending`.
    "lg_conv3x3s1p1",
    // 1x1 conv. NAFBlock is built out of four of them per block (conv1, conv3,
    // sca, conv4, conv5), so this is the model's most-called kernel by a wide
    // margin.
    "lg_conv1x1",
    // The per-channel NCHW LayerNorm. `LayerNorm2d` in the reference is
    // `F.layer_norm` over the channel axis with per-channel weight and bias and
    // eps 1e-6, which is exactly this op's contract.
    "lg_channel_layer_norm",
    // The channel attention's global average pool: `sca` averages each channel
    // over the whole feature map before its 1x1 conv.
    "lg_channel_mean",
    // SimpleGate is `x.chunk(2, dim=1)` then multiply - the op the gated
    // architectures need on their own terms, and the reason this kernel is in
    // the toolkit at all.
    "lg_mul",
    // The channel attention's `x * sca(x)`: a per-channel scale of an NCHW
    // tensor by a `[c]` activation.
    "lg_channel_scale",
    "lg_add",
    "lg_add_scaled",
];

/// This project's own kernels, in `cuda/nafnet.cu`. Both are ops the toolkit
/// does not have: a grouped/depthwise convolution (`conv2` is 3x3 with
/// `groups = 2*width`, one weight set per channel) and the DEPTH-TO-SPACE
/// direction of PixelShuffle. The toolkit's `lg_pixel_unshuffle2` is the
/// opposite direction, so neither is a duplicate of anything shared.
const PROJECT_KERNELS: &[&str] = &[
    "nf_conv3x3_dw",
    "nf_pixel_shuffle2",
    "nf_down2x2s2",
    "nf_conv1x1_oc",
    "nf_residual",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/nafnet.cu");

    // `cargo build --no-default-features` is the pure-Rust CPU build: it must
    // not need nvcc, and `src/cuda.rs` (which includes the fatbins) is not
    // compiled at all, so the env vars it would embed are not needed.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/nafnet.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/nafnet.cu").expect("read cuda/nafnet.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/nafnet.cu (it has {})",
            defined.join(", ")
        );
    }
    // The other direction matters just as much: a kernel defined but NOT listed
    // is pruned from the fatbin by `--entries`, and then it fails at LAUNCH
    // rather than at build time. Checking both directions makes the list and the
    // source agree rather than merely overlap.
    for d in &defined {
        assert!(
            PROJECT_KERNELS.contains(&d.as_str()),
            "cuda/nafnet.cu defines `{d}`, which PROJECT_KERNELS does not list - \
             it would be pruned from the fatbin and fail at launch"
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "nafnet_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/nafnet.cu",
            out_name: "nafnet_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}

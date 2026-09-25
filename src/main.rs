//! NAFNet image restoration (deblur, denoise), standalone.
//!
//! Usage:
//!   nafnet --model nafnet-gopro-width32.safetensors -i in.png -o out.png
//!   nafnet --model ... -i in.png --device cpu
//!
//! The model file is the output of `tools/convert.py`, not the original .pth.

#[cfg(feature = "cuda")]
mod cuda;
mod image;
mod net;
mod weights;
#[cfg(feature = "cuda")]
mod gpu;

use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> ! {
    eprintln!(
        "nafnet {VERSION} - NAFNet image restoration on lightgpu

USAGE:
    nafnet --model <weights.safetensors> -i <in.png> -o <out.png> [options]

OPTIONS:
    -m, --model <path>    converted .safetensors checkpoint (see tools/convert.py)
    -i, --input <path>    input PNG, or - for stdin (default: stdin)
    -o, --output <path>   output PNG, or - for stdout (default: stdout)
        --device <dev>    gpu or cpu (default: the CUDA build uses gpu, a
                          CPU-only build uses cpu)
        --pad <mode>      how the input is padded up to a multiple of 16:
                          `reflect` (the default and what the reference uses) or
                          `zero`
    -q, --quiet           no progress output
    -h, --help            this text
    -V, --version         print the version"
    );
    // THE DEVELOPMENT FLAGS ARE LISTED ONLY BY A BINARY THAT HAS THEM. Help text
    // that advertises an option the parser rejects is worse than no help, and the
    // reverse mistake is how `--profile` spent its whole life inert while
    // `--help` promised a per-kernel profile that did not exist.
    #[cfg(feature = "dev")]
    eprintln!(
        "
DEVELOPMENT ONLY (this build has `--features dev`; a release build has none of
these, and rejects them by name):
        --cuda-selftest   compare each CUDA kernel against its CPU twin, at real
                          sizes and a relative tolerance, and exit
        --profile         report per-kernel GPU time, longest first (CUDA event
                          pairs around every launch). On the CPU path, set
                          NAFNET_CPU_PROFILE=1 for the per-op table - there is no
                          host clock that can attribute time to an asynchronous
                          launch.
        --size <h> <w>    the geometry of a --raw plane
        --raw <path>      read a preprocessed input plane instead of an image:
                          [3][h][w] f32, little-endian, with h and w from
                          --size. Paired with tools/reference.py's --dump/--seed
                          it gives BOTH sides byte-identical input, which is the
                          only way a per-stage comparison is meaningful.
        --dump <path>     write every intermediate activation as a flat f32 file
                          plus a text index of `<name> <c> <h> <w> <offset>`
                          lines. BOTH backends implement it, and on the GPU it
                          costs a device-to-host copy per stage."
    );
    std::process::exit(2)
}

/// Pad `img` to a multiple of `mult`, returning the padded image and the
/// original size. The reference uses `F.pad(x, (0, w%mult, 0, h%mult),
/// mode='reflect')`, and reflection is not cosmetic: zero-padding changes the
/// first/last rows the network sees and therefore the output near the border.
fn pad_to(img: &image::Image, mult: usize, mode: &str) -> (image::Image, usize, usize) {
    let (h, w) = (img.h, img.w);
    if h % mult == 0 && w % mult == 0 {
        return (image::Image { w, h, data: img.data.clone() }, h, w);
    }
    let oh = h.div_ceil(mult) * mult;
    let ow = w.div_ceil(mult) * mult;
    let mut out = image::Image::new(ow, oh);
    for c in 0..3 {
        let src = img.plane(c);
        let dst = &mut out.data[c * ow * oh..(c + 1) * ow * oh];
        for y in 0..oh {
            for x in 0..ow {
                let v = if y < h && x < w {
                    src[y * w + x]
                } else {
                    match mode {
                        "zero" => 0.0,
                        _ => {
                            // Reflect without repeating the edge sample, which is
                            // what torch's mode='reflect' does.
                            let r = |i: isize, n: isize| -> isize {
                                if n == 1 {
                                    0
                                } else {
                                    let p = 2 * n - 2;
                                    let mut v = i.rem_euclid(p);
                                    if v >= n {
                                        v = p - v;
                                    }
                                    v
                                }
                            };
                            let yy = r(y as isize, h as isize) as usize;
                            let xx = r(x as isize, w as isize) as usize;
                            src[yy * w + xx]
                        }
                    }
                };
                dst[y * ow + x] = v;
            }
        }
    }
    (out, h, w)
}

fn crop(out: &[f32], c: usize, ow: usize, oh: usize, w: usize, h: usize) -> image::Image {
    let mut img = image::Image::new(w, h);
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                img.data[ch * w * h + y * w + x] = out[ch * ow * oh + y * ow + x];
            }
        }
    }
    img
}

/// Write a dump: the flat little-endian f32 blob plus a text index of
/// `<name> <c> <h> <w> <offset>` lines.
///
/// DEVELOPMENT ONLY. The index is the part that matters and the reason both
/// sides of a comparison are written this way: it carries the stage NAME, so a
/// divergence is found by name rather than by comparing two dumps at identical
/// byte offsets - which is wrong the moment the two backends dump a different
/// set of stages.
#[cfg(feature = "dev")]
fn write_dump(path: &str, index: &[String], blob: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(blob.len() * 4);
    for v in blob {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())?;
    std::fs::write(format!("{path}.txt"), index.join("\n") + "\n").map_err(|e| e.to_string())?;
    Ok(())
}

/// The device dispatch for the `--raw` path.
///
/// DEVELOPMENT ONLY: `--raw` is the only caller, so a release build does not
/// carry this at all.
#[cfg(feature = "dev")]
fn run_backend(
    weights: &weights::Weights,
    geo: net::Geometry,
    input_plane: &[f32],
    h: usize,
    w: usize,
    device: &str,
    dump_path: Option<&str>,
    prof: bool,
) -> Result<Vec<f32>, String> {
    match device {
        #[cfg(feature = "cuda")]
        "gpu" => {
            // THE PROFILE BELONGS HERE AS MUCH AS ON THE IMAGE PATH: this is the
            // branch the `--raw`/`--size` entry point runs, and `--raw` is how a
            // size other than a real photograph's is measured at all.
            let g = gpu::Gpu::with_profile(weights, geo, prof)?;
            let v = match dump_path {
                None => g.forward(input_plane, h, w)?,
                Some(p) => {
                    let mut index: Vec<String> = Vec::new();
                    let mut blob: Vec<f32> = Vec::new();
                    let v = g.forward_dump(input_plane, h, w, &mut |name, c, h, w, data| {
                        let off = blob.len();
                        blob.extend_from_slice(data);
                        index.push(format!("{name} {c} {h} {w} {off}"));
                    })?;
                    write_dump(p, &index, &blob)?;
                    v
                }
            };
            if let Some(p) = &g.profile {
                p.report();
            }
            Ok(v)
        }
        // A CPU-only build has no GPU branch, and `geo` and `prof` are only
        // meaningful there - naming them keeps this from reading as unused
        // parameters rather than as an absent backend.
        #[cfg(not(feature = "cuda"))]
        "gpu" => {
            let _ = (geo, prof);
            Err("this build has no cuda feature; use --device cpu".into())
        }
        "cpu" => match dump_path {
            None => net::forward_cpu(weights, input_plane, h, w).map(|a| a.data),
            Some(p) => {
                let mut index: Vec<String> = Vec::new();
                let mut blob: Vec<f32> = Vec::new();
                let a = net::forward_cpu_dump(weights, input_plane, h, w, &mut |name: &str, act: &net::Act| {
                    let off = blob.len();
                    blob.extend_from_slice(&act.data);
                    index.push(format!("{name} {} {} {} {}", act.c, act.h, act.w, off));
                })?;
                write_dump(p, &index, &blob)?;
                Ok(a.data)
            }
        },
        other => Err(format!("unknown device `{other}` (gpu or cpu)")),
    }
}

/// One GPU pass with an optional stage dump.
///
/// THE SINGLE CALL PATH FOR BOTH BUILDS, so the release and the development
/// binary run the same code and only the ARGUMENTS differ: a release build
/// reaches `dump_path == None` because no flag can set it, not because a
/// different function was compiled. That is what makes "the release binary is
/// the same engine" checkable.
#[cfg(feature = "cuda")]
fn forward_gpu(
    g: &gpu::Gpu,
    plane: &[f32],
    h: usize,
    w: usize,
    dump_path: Option<&str>,
) -> Result<Vec<f32>, String> {
    match dump_path {
        None => g.forward(plane, h, w),
        #[cfg(feature = "dev")]
        Some(p) => {
            let mut index: Vec<String> = Vec::new();
            let mut blob: Vec<f32> = Vec::new();
            let r = g.forward_dump(plane, h, w, &mut |name, c, h, w, data| {
                let off = blob.len();
                blob.extend_from_slice(data);
                index.push(format!("{name} {c} {h} {w} {off}"));
            });
            if r.is_ok() {
                write_dump(p, &index, &blob)?;
            }
            r
        }
        // Unreachable in a release build: no flag can set a dump path there.
        #[cfg(not(feature = "dev"))]
        Some(_) => g.forward(plane, h, w),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut model_path: Option<String> = None;
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut device = if cfg!(feature = "cuda") { "gpu" } else { "cpu" }.to_string();
    let mut pad_mode = "reflect".to_string();
    let mut quiet = false;
    // DEV-ONLY STATE, AND ITS ABSENCE IS WHAT KEEPS THE FLAGS OUT. A release
    // build has no flag that sets any of these, so the variables do not exist in
    // it - which also means a later edit cannot re-expose a development flag by
    // wiring it to state that is still being parsed.
    #[cfg(feature = "dev")]
    let mut dump_path: Option<String> = None;
    #[cfg(feature = "dev")]
    let mut raw_path: Option<String> = None;
    #[cfg(feature = "dev")]
    let mut size: Option<(usize, usize)> = None;
    #[cfg(feature = "dev")]
    let mut cuda_selftest = false;
    // A RELEASE BUILD HAS NO FLAG THAT SETS THIS, so for the GPU branch it is not
    // a runtime choice - it is the constant the compiler folds into the launch
    // path. PROFILING IS ALSO A GPU CONCERN: there is no launch to time in a
    // CPU-only build, so the binding does not exist there at all.
    #[cfg(all(not(feature = "dev"), feature = "cuda"))]
    let profile = false;
    #[cfg(feature = "dev")]
    let mut profile = false;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| usage())
        };
        match a {
            "-m" | "--model" => model_path = Some(next(&mut i)),
            "-i" | "--input" => input = Some(next(&mut i)),
            "-o" | "--output" => output = Some(next(&mut i)),
            "--device" => device = next(&mut i),
            "--pad" => pad_mode = next(&mut i),
            "-q" | "--quiet" => quiet = true,
            // THE DEVELOPMENT FLAGS, AND ONLY A DEVELOPMENT BUILD HAS THEM. A
            // release build refuses them by name rather than ignoring them, which
            // matters most for `--dump`: a script that asked for a dump and did
            // not get one must not carry on as if it had.
            #[cfg(feature = "dev")]
            "--dump" => dump_path = Some(next(&mut i)),
            #[cfg(feature = "dev")]
            "--raw" => raw_path = Some(next(&mut i)),
            #[cfg(feature = "dev")]
            "--size" => {
                let h: usize = next(&mut i).parse().unwrap_or(0);
                let w: usize = next(&mut i).parse().unwrap_or(0);
                size = Some((h, w));
            }
            #[cfg(feature = "dev")]
            "--cuda-selftest" => cuda_selftest = true,
            #[cfg(feature = "dev")]
            "--profile" => profile = true,
            // `--tile` IS NOT A DEVELOPMENT FLAG, IT IS NOT AN OPTION AT ALL.
            // Other engines in the family take it, so a driver script shared
            // across them may pass it - and a flag that is silently accepted and
            // ignored would let a caller believe it had bounded the memory this
            // process will use. Answer with the reason instead of a fake success.
            "--tile" | "--tile-pad" => {
                eprintln!("nafnet: `{a}` is not supported: this engine runs a whole image at once");
                eprintln!("nafnet: (NAFNet is fully convolutional; nothing here tiles or needs to)");
                std::process::exit(2);
            }
            // THE DEVELOPMENT FLAGS ARE IN NO BUILD BUT A `dev` ONE. Refused by
            // name rather than ignored - a script that asked for a dump and did
            // not get one must not carry on as if it had.
            #[cfg(not(feature = "dev"))]
            "--dump" | "--raw" | "--size" | "--cuda-selftest" | "--profile" => {
                eprintln!("nafnet: `{a}` is a development flag and this is a release build");
                eprintln!("nafnet: rebuild with `cargo build --release --features dev` for --dump,");
                eprintln!("nafnet: --raw, --size, --cuda-selftest and --profile");
                std::process::exit(2);
            }
            "-h" | "--help" => usage(),
            "-V" | "--version" => {
                println!("nafnet {VERSION}");
                return;
            }
            other => {
                eprintln!("nafnet: unknown argument `{other}`");
                usage();
            }
        }
        i += 1;
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("nafnet: --model is required (see tools/convert.py)");
        usage()
    });
    let weights = match weights::Weights::open(&model_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("nafnet: {e}");
            std::process::exit(1);
        }
    };
    let cfg = weights.config.clone();
    if !quiet {
        eprintln!(
            "nafnet {VERSION}: width {} levels {} enc {:?} mid {} dec {:?} (task {})",
            cfg.width,
            cfg.levels(),
            cfg.enc_blk_nums,
            cfg.middle_blk_num,
            cfg.dec_blk_nums,
            cfg.task
        );
    }
    let geo = net::Geometry {
        // The `cuda`-gated fields are the ones only the GPU executor reads; see
        // `net::Geometry`.
        #[cfg(feature = "cuda")]
        width: cfg.width,
        enc_blk_nums: cfg.enc_blk_nums.clone(),
        #[cfg(feature = "cuda")]
        middle_blk_num: cfg.middle_blk_num,
        #[cfg(feature = "cuda")]
        dec_blk_nums: cfg.dec_blk_nums.clone(),
    };
    let mult = geo.padder_size();

    #[cfg(all(feature = "cuda", feature = "dev"))]
    if cuda_selftest {
        let g = match gpu::Gpu::with_profile(&weights, geo.clone(), false) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("nafnet: {e}");
                std::process::exit(1);
            }
        };
        match g.selftest() {
            Ok((fails, rows)) => {
                for r in &rows {
                    println!("{r}");
                }
                println!(
                    "{} ops checked, {} failed{}",
                    rows.len(),
                    fails,
                    if fails == 0 { "" } else { " - see the list above" }
                );
                std::process::exit(if fails == 0 { 0 } else { 1 });
            }
            Err(e) => {
                eprintln!("nafnet: {e}");
                std::process::exit(1);
            }
        }
    }
    #[cfg(all(not(feature = "cuda"), feature = "dev"))]
    if cuda_selftest {
        eprintln!("nafnet: --cuda-selftest needs a build with the cuda feature");
        std::process::exit(2);
    }

    // A raw plane bypasses the image path entirely: [3][h][w] f32, little
    // endian, with h and w from `--size`. This exists so the engine can be fed
    // the exact tensor the reference recorded, which is the only way a
    // stage-by-stage comparison is comparing the MODEL rather than the loader.
    #[cfg(feature = "dev")]
    if let Some(p) = raw_path.as_deref() {
        let (h, w) = match size {
            Some(v) => v,
            None => {
                eprintln!("nafnet: --raw needs --size h w (a raw plane has no header)");
                std::process::exit(2);
            }
        };
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("nafnet: read {p}: {e}");
                std::process::exit(1);
            }
        };
        if bytes.len() != 3 * h * w * 4 {
            eprintln!(
                "nafnet: {p} is {} bytes but --size {h} {w} needs {}",
                bytes.len(),
                3 * h * w * 4
            );
            std::process::exit(2);
        }
        let mut plane = Vec::with_capacity(3 * h * w);
        for ch in bytes.chunks_exact(4) {
            plane.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
        }
        // Pad exactly as the image path does, so the compared tensors have the
        // geometry the reference's padded input has.
        let img = image::Image { w, h, data: plane };
        let (padded, ph, pw) = pad_to(&img, mult, &pad_mode);
        let t0 = Instant::now();
        let out = match run_backend(&weights, geo, &padded.data, ph, pw, &*device, dump_path.as_deref(), profile) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("nafnet: {e}");
                std::process::exit(1);
            }
        };
        let cropped = crop(&out, 3, padded.w, padded.h, w, h);
        if !quiet {
            eprintln!("nafnet: {w}x{h} -> {w}x{h} in {:.2}s", t0.elapsed().as_secs_f32());
        }
        // A raw plane writes raw: the caller compares numbers, not pixels.
        if let Some(p) = output.as_deref() {
            if p != "-" {
                let mut bytes = Vec::with_capacity(cropped.data.len() * 4);
                for v in &cropped.data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                if let Err(e) = std::fs::write(p, bytes) {
                    eprintln!("nafnet: write {p}: {e}");
                    std::process::exit(1);
                }
                return;
            }
        }
        let rgb = cropped.to_rgb8();
        {
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            if let Err(e) = image::save_rgb_stream(&mut lock, w, h, &rgb).and_then(|_| lock.flush().map_err(|e| e.to_string())) {
                eprintln!("nafnet: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // The image. `-` or absent means stdin.
    let img = match input.as_deref() {
        None | Some("-") => {
            use std::io::Read;
            let mut buf = Vec::new();
            if let Err(e) = std::io::stdin().read_to_end(&mut buf) {
                eprintln!("nafnet: read stdin: {e}");
                std::process::exit(1);
            }
            image::load_rgb_stream(&buf[..])
        }
        Some(p) => image::load_rgb(p),
    };
    let img = match img {
        Ok(v) => v,
        Err(e) => {
            eprintln!("nafnet: {e}");
            std::process::exit(1);
        }
    };

    let t0 = Instant::now();

    // THE WHOLE IMAGE, ALWAYS. NAFNet is fully convolutional and the released
    // test configurations run at native resolution, and this engine has no tile
    // path at all - a caller who passes `--tile` is told so rather than getting
    // silence. Memory is bounded by the plan instead: see `gpu::Plan`.
    let (padded, h, w) = pad_to(&img, mult, &pad_mode);
    let input_plane = padded.data.clone();

    let out: Vec<f32> = match device.as_str() {
        #[cfg(feature = "cuda")]
        "gpu" => {
            // A RELEASE BUILD CANNOT EVEN ASK FOR A PROFILE OR A DUMP: the
            // arguments do not exist there, so both are the const `false`/`None`
            // rather than runtime flags, and the profile struct is never built.
            let g = match gpu::Gpu::with_profile(&weights, geo.clone(), profile) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("nafnet: {e}");
                    std::process::exit(1);
                }
            };
            if !quiet {
                eprintln!("nafnet: device {}", g.device_name());
            }
            #[cfg(feature = "dev")]
            let dumped = dump_path.as_deref();
            #[cfg(not(feature = "dev"))]
            let dumped: Option<&str> = None;
            let v = match forward_gpu(&g, &input_plane, padded.h, padded.w, dumped) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("nafnet: {e}");
                    std::process::exit(1);
                }
            };
            // AND THE REPORT IS GATED TOO: profiling is only reachable with the
            // development flags, so a release build does not link the table.
            #[cfg(feature = "dev")]
            if let Some(p) = &g.profile {
                p.report();
            }
            v
        }
        #[cfg(not(feature = "cuda"))]
        "gpu" => {
            eprintln!("nafnet: this build has no cuda feature; use --device cpu");
            std::process::exit(2);
        }
        "cpu" => {
            #[cfg(feature = "dev")]
            let r = net::forward_cpu_maybe_dump(&weights, &input_plane, padded.h, padded.w, dump_path.as_deref());
            #[cfg(not(feature = "dev"))]
            let r = net::forward_cpu(&weights, &input_plane, padded.h, padded.w);
            match r {
                Ok(a) => a.data,
                Err(e) => {
                    eprintln!("nafnet: {e}");
                    std::process::exit(1);
                }
            }
        }
        other => {
            eprintln!("nafnet: unknown device `{other}` (gpu or cpu)");
            std::process::exit(2);
        }
    };

    let cropped = crop(&out, 3, padded.w, padded.h, w, h);
    if !quiet {
        eprintln!(
            "nafnet: {}x{} -> {}x{} in {:.2}s",
            w, h, w, h,
            t0.elapsed().as_secs_f32()
        );
    }
    let rgb = cropped.to_rgb8();
    let res = match output.as_deref() {
        None | Some("-") => {
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            image::save_rgb_stream(&mut lock, w, h, &rgb).and_then(|_| {
                lock.flush().map_err(|e| e.to_string())
            })
        }
        Some(p) => image::save_rgb(p, w, h, &rgb),
    };
    if let Err(e) = res {
        eprintln!("nafnet: {e}");
        std::process::exit(1);
    }
}

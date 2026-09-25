//! Weights: the NAFNet reading of a converted `.safetensors` checkpoint.
//!
//! The container itself - `mmap`, the JSON header, tensor offsets and borrowed
//! slices - is `lightgpu::safetensors`, shared with every other engine in the
//! family. What is left here is what is specific to NAFNet:
//!
//! * the architecture constants (`width`, `enc_blk_nums`, `middle_blk_num`,
//!   `dec_blk_nums`, `task`) are read out of `__metadata__`, because there are
//!   four published shapes in the wild and the file is the only place that
//!   knows which one it is;
//! * every tensor the graph walks is shape-checked against those constants at
//!   load. NAFNet is unusually easy to get subtly wrong - a `groups=2*c`
//!   depthwise conv whose weight is silently read as a dense conv produces a
//!   plausible image from the wrong arithmetic - so the shapes are asserted
//!   here rather than discovered in the output;
//! * `get()` returns a zero-copy `&[f32]` into the mapping.
use lightgpu::safetensors::File;

/// The architecture constants the engine needs before it can walk the graph.
///
/// `enc_blk_nums` and `dec_blk_nums` are stored as a comma-separated list in
/// the metadata (JSON cannot hold a list in a safetensors `__metadata__` map,
/// which is string-to-string).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub width: usize,
    pub enc_blk_nums: Vec<usize>,
    pub middle_blk_num: usize,
    pub dec_blk_nums: Vec<usize>,
    /// The dataset the checkpoint was trained on, purely informational: it is
    /// what the accuracy figures in the README are quoted against.
    pub task: String,
}

impl Config {
    /// Levels in the U-Net. The encoders and decoders are the same length in
    /// every released config.
    pub fn levels(&self) -> usize {
        self.enc_blk_nums.len()
    }

    /// Feature width at level `l`, before the block stack. The encoder doubles
    /// per level and the decoder halves back.
    pub fn width_at(&self, level: usize) -> usize {
        self.width << level
    }
}

pub struct Weights {
    pub file: File,
    pub config: Config,
}

fn parse_nums(s: &str, what: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|_| format!("`{what}` is not a comma-separated list of numbers: `{s}`"))
        })
        .collect()
}

impl Weights {
    pub fn open(path: &str) -> Result<Weights, String> {
        let file = File::open(path).map_err(|e| format!("{e} (not a converted NAFNet file?)"))?;
        let get = |k: &str| -> Result<String, String> {
            file.metadata_get(k)
                .map(|v| v.to_string())
                .ok_or_else(|| {
                    format!("checkpoint metadata is missing `{k}` (not a converted NAFNet file?)")
                })
        };
        let config = Config {
            width: get("width")?
                .parse()
                .map_err(|_| "metadata `width` is not a number".to_string())?,
            enc_blk_nums: parse_nums(&get("enc_blk_nums")?, "enc_blk_nums")?,
            middle_blk_num: get("middle_blk_num")?
                .parse()
                .map_err(|_| "metadata `middle_blk_num` is not a number".to_string())?,
            dec_blk_nums: parse_nums(&get("dec_blk_nums")?, "dec_blk_nums")?,
            task: get("task").unwrap_or_else(|_| "unknown".into()),
        };
        if config.levels() == 0 {
            return Err("metadata `enc_blk_nums` is empty".into());
        }
        if config.enc_blk_nums.len() != config.dec_blk_nums.len() {
            return Err(format!(
                "enc_blk_nums has {} levels but dec_blk_nums has {}",
                config.enc_blk_nums.len(),
                config.dec_blk_nums.len()
            ));
        }
        let w = Weights { file, config };
        w.check_shapes()?;
        Ok(w)
    }

    pub fn get(&self, name: &str) -> Result<&[f32], String> {
        self.file.f32(name)
    }

    /// `conv.weight` with an explicit expected shape. Every tensor the graph
    /// reads goes through one of these, so a checkpoint from the wrong config
    /// fails at load rather than at the first conv.
    fn conv(&self, name: &str, oc: usize, ic: usize, k: usize) -> Result<(), String> {
        let got = self.get(name)?;
        let want = oc * ic * k * k;
        if got.len() != want {
            return Err(format!(
                "{name}: expected [{oc}, {ic}, {k}, {k}] = {want} values, found {}",
                got.len()
            ));
        }
        Ok(())
    }

    fn vec1(&self, name: &str, n: usize) -> Result<(), String> {
        let got = self.get(name)?;
        if got.len() != n {
            return Err(format!("{name}: expected {n} values, found {}", got.len()));
        }
        Ok(())
    }

    /// One NAFBlock's tensors. `c` is the block's channel count, DW_Expand and
    /// FFN_Expand are both 2 in every released config.
    fn block(&self, prefix: &str, c: usize) -> Result<(), String> {
        let dw = c * 2;
        self.conv(&format!("{prefix}.conv1.weight"), dw, c, 1)?;
        self.vec1(&format!("{prefix}.conv1.bias"), dw)?;
        // Depthwise: groups == dw, so the weight is [dw, 1, 3, 3].
        self.conv(&format!("{prefix}.conv2.weight"), dw, 1, 3)?;
        self.vec1(&format!("{prefix}.conv2.bias"), dw)?;
        self.conv(&format!("{prefix}.conv3.weight"), c, dw / 2, 1)?;
        self.vec1(&format!("{prefix}.conv3.bias"), c)?;
        // The channel attention: a 1x1 conv over the POST-SimpleGate channels
        // applied to the pooled map, so it is [dw/2, dw/2, 1, 1].
        self.conv(&format!("{prefix}.sca.1.weight"), dw / 2, dw / 2, 1)?;
        self.vec1(&format!("{prefix}.sca.1.bias"), dw / 2)?;
        let ffn = c * 2;
        self.conv(&format!("{prefix}.conv4.weight"), ffn, c, 1)?;
        self.vec1(&format!("{prefix}.conv4.bias"), ffn)?;
        self.conv(&format!("{prefix}.conv5.weight"), c, ffn / 2, 1)?;
        self.vec1(&format!("{prefix}.conv5.bias"), c)?;
        self.vec1(&format!("{prefix}.norm1.weight"), c)?;
        self.vec1(&format!("{prefix}.norm1.bias"), c)?;
        self.vec1(&format!("{prefix}.norm2.weight"), c)?;
        self.vec1(&format!("{prefix}.norm2.bias"), c)?;
        self.vec1(&format!("{prefix}.beta"), c)?;
        self.vec1(&format!("{prefix}.gamma"), c)?;
        Ok(())
    }

    fn check_shapes(&self) -> Result<(), String> {
        let cfg = &self.config;
        self.conv("intro.weight", cfg.width, 3, 3)?;
        self.vec1("intro.bias", cfg.width)?;
        self.conv("ending.weight", 3, cfg.width, 3)?;
        self.vec1("ending.bias", 3)?;

        for l in 0..cfg.levels() {
            let c = cfg.width_at(l);
            for b in 0..cfg.enc_blk_nums[l] {
                self.block(&format!("encoders.{l}.{b}"), c)?;
            }
            // The downsample is a stride-2 2x2 conv that doubles the channels.
            self.conv(&format!("downs.{l}.weight"), 2 * c, c, 2)?;
            self.vec1(&format!("downs.{l}.bias"), 2 * c)?;
        }
        let mid = cfg.width << cfg.levels();
        for b in 0..cfg.middle_blk_num {
            self.block(&format!("middle_blks.{b}"), mid)?;
        }
        // The upsample is a 1x1 conv then PixelShuffle(2); the conv is BIAS-FREE
        // in the reference (`nn.Conv2d(chan, chan*2, 1, bias=False)`), which is
        // easy to miss and would otherwise look like a missing tensor.
        let mut chan = mid;
        for l in 0..cfg.levels() {
            self.conv(&format!("ups.{l}.0.weight"), chan * 2, chan, 1)?;
            chan /= 2;
            for b in 0..cfg.dec_blk_nums[l] {
                self.block(&format!("decoders.{l}.{b}"), chan)?;
            }
        }
        if chan != cfg.width {
            return Err(format!("decoder converges to width {chan}, expected {}", cfg.width));
        }
        Ok(())
    }
}

//! Host CPU feature detection, ISA-level capping, and Cranelift ISA configuration.
//!
//! Feature names use LLVM spelling (`avx512f`, `amx-tile`, `sve2`) so the same strings can be
//! passed to `--target-features` and, later, to an LLVM target machine.
//!
//! An ISA cap (`--isa`, `ACHAINSAW_MAX_ISA`, or [`set_isa_cap`]) hides every feature above a
//! level, which lets one machine exercise all lower tiers (e.g. AVX2 code on an AVX-512 host).

use anyhow::{anyhow, Result};
use cranelift_codegen::isa;
use cranelift_codegen::settings::Configurable;
use serde_json::{json, Value};
use std::fmt;
use std::str::FromStr;
use std::sync::{OnceLock, RwLock};

/// Environment variable that caps the ISA level used for code generation.
pub const ISA_CAP_ENV: &str = "ACHAINSAW_MAX_ISA";

/// Vector register width Cranelift generates code for, regardless of host features.
pub const CRANELIFT_VECTOR_BITS: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Aarch64,
    Other,
}

impl Arch {
    pub fn host() -> Self {
        if cfg!(target_arch = "x86_64") {
            Arch::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else {
            Arch::Other
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
            Arch::Other => "other",
        }
    }
}

/// Vector ISA tiers, ordered within each architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaLevel {
    /// x86-64-v2: SSE3 through SSE4.2, POPCNT, CMPXCHG16B.
    Sse,
    /// AVX, F16C.
    Avx,
    /// x86-64-v3: AVX2, FMA, BMI1/2, LZCNT, AVX-VNNI.
    Avx2,
    /// x86-64-v4 and later AVX-512 extensions.
    Avx512,
    /// AMX tile matrix engine.
    Amx,
    /// AArch64 baseline: NEON plus scalar/atomic extensions.
    Neon,
    Sve,
    Sve2,
    /// SME/SME2 streaming matrix engine.
    Sme,
}

const ALL_LEVELS: &[IsaLevel] = &[
    IsaLevel::Sse,
    IsaLevel::Avx,
    IsaLevel::Avx2,
    IsaLevel::Avx512,
    IsaLevel::Amx,
    IsaLevel::Neon,
    IsaLevel::Sve,
    IsaLevel::Sve2,
    IsaLevel::Sme,
];

impl IsaLevel {
    pub fn arch(&self) -> Arch {
        match self {
            IsaLevel::Sse | IsaLevel::Avx | IsaLevel::Avx2 | IsaLevel::Avx512 | IsaLevel::Amx => {
                Arch::X86_64
            }
            IsaLevel::Neon | IsaLevel::Sve | IsaLevel::Sve2 | IsaLevel::Sme => Arch::Aarch64,
        }
    }

    /// Position within the architecture family; higher includes everything below.
    fn rank(&self) -> u8 {
        match self {
            IsaLevel::Sse | IsaLevel::Neon => 0,
            IsaLevel::Avx | IsaLevel::Sve => 1,
            IsaLevel::Avx2 | IsaLevel::Sve2 => 2,
            IsaLevel::Avx512 | IsaLevel::Sme => 3,
            IsaLevel::Amx => 4,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            IsaLevel::Sse => "sse",
            IsaLevel::Avx => "avx",
            IsaLevel::Avx2 => "avx2",
            IsaLevel::Avx512 => "avx512",
            IsaLevel::Amx => "amx",
            IsaLevel::Neon => "neon",
            IsaLevel::Sve => "sve",
            IsaLevel::Sve2 => "sve2",
            IsaLevel::Sme => "sme",
        }
    }

    pub fn names() -> Vec<&'static str> {
        ALL_LEVELS.iter().map(|l| l.as_str()).collect()
    }
}

impl fmt::Display for IsaLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for IsaLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        ALL_LEVELS
            .iter()
            .copied()
            .find(|l| l.as_str() == lower)
            .ok_or_else(|| {
                format!(
                    "unknown ISA level '{s}'; expected one of: {}",
                    IsaLevel::names().join(", ")
                )
            })
    }
}

/// A detectable CPU feature. Discriminants index bits in [`CpuFeatures`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Feature {
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
    Popcnt,
    Cx16,
    Avx,
    F16c,
    Avx2,
    Fma,
    Bmi1,
    Bmi2,
    Lzcnt,
    AvxVnni,
    Avx512f,
    Avx512vl,
    Avx512bw,
    Avx512dq,
    Avx512cd,
    Avx512bf16,
    Avx512fp16,
    Avx512vnni,
    Avx512bitalg,
    Avx512vbmi,
    AmxTile,
    AmxBf16,
    AmxInt8,
    AmxFp16,
    Neon,
    Fp16,
    Dotprod,
    I8mm,
    Lse,
    Pauth,
    Sve,
    Sve2,
    Sme,
    Sme2,
}

struct FeatureInfo {
    feature: Feature,
    /// LLVM target-feature spelling.
    name: &'static str,
    level: IsaLevel,
    /// Cranelift ISA setting enabled by this feature, if Cranelift uses it.
    cranelift_flag: Option<&'static str>,
}

const fn info(
    feature: Feature,
    name: &'static str,
    level: IsaLevel,
    cranelift_flag: Option<&'static str>,
) -> FeatureInfo {
    FeatureInfo {
        feature,
        name,
        level,
        cranelift_flag,
    }
}

use Feature as F;
use IsaLevel as L;

const FEATURE_TABLE: &[FeatureInfo] = &[
    info(F::Sse3, "sse3", L::Sse, Some("has_sse3")),
    info(F::Ssse3, "ssse3", L::Sse, Some("has_ssse3")),
    info(F::Sse41, "sse4.1", L::Sse, Some("has_sse41")),
    info(F::Sse42, "sse4.2", L::Sse, Some("has_sse42")),
    info(F::Popcnt, "popcnt", L::Sse, Some("has_popcnt")),
    info(F::Cx16, "cx16", L::Sse, Some("has_cmpxchg16b")),
    info(F::Avx, "avx", L::Avx, Some("has_avx")),
    info(F::F16c, "f16c", L::Avx, None),
    info(F::Avx2, "avx2", L::Avx2, Some("has_avx2")),
    info(F::Fma, "fma", L::Avx2, Some("has_fma")),
    info(F::Bmi1, "bmi", L::Avx2, Some("has_bmi1")),
    info(F::Bmi2, "bmi2", L::Avx2, Some("has_bmi2")),
    info(F::Lzcnt, "lzcnt", L::Avx2, Some("has_lzcnt")),
    info(F::AvxVnni, "avxvnni", L::Avx2, Some("has_avx_vnni")),
    info(F::Avx512f, "avx512f", L::Avx512, Some("has_avx512f")),
    info(F::Avx512vl, "avx512vl", L::Avx512, Some("has_avx512vl")),
    info(F::Avx512bw, "avx512bw", L::Avx512, None),
    info(F::Avx512dq, "avx512dq", L::Avx512, Some("has_avx512dq")),
    info(F::Avx512cd, "avx512cd", L::Avx512, None),
    info(F::Avx512bf16, "avx512bf16", L::Avx512, None),
    info(F::Avx512fp16, "avx512fp16", L::Avx512, None),
    info(
        F::Avx512vnni,
        "avx512vnni",
        L::Avx512,
        Some("has_avx512vnni"),
    ),
    info(
        F::Avx512bitalg,
        "avx512bitalg",
        L::Avx512,
        Some("has_avx512bitalg"),
    ),
    info(
        F::Avx512vbmi,
        "avx512vbmi",
        L::Avx512,
        Some("has_avx512vbmi"),
    ),
    info(F::AmxTile, "amx-tile", L::Amx, None),
    info(F::AmxBf16, "amx-bf16", L::Amx, None),
    info(F::AmxInt8, "amx-int8", L::Amx, None),
    info(F::AmxFp16, "amx-fp16", L::Amx, None),
    info(F::Neon, "neon", L::Neon, None),
    info(F::Fp16, "fullfp16", L::Neon, Some("has_fp16")),
    info(F::Dotprod, "dotprod", L::Neon, Some("has_dotprod")),
    info(F::I8mm, "i8mm", L::Neon, Some("has_i8mm")),
    info(F::Lse, "lse", L::Neon, Some("has_lse")),
    info(F::Pauth, "pauth", L::Neon, Some("has_pauth")),
    info(F::Sve, "sve", L::Sve, None),
    info(F::Sve2, "sve2", L::Sve2, None),
    info(F::Sme, "sme", L::Sme, None),
    info(F::Sme2, "sme2", L::Sme, None),
];

fn feature_info(f: Feature) -> &'static FeatureInfo {
    &FEATURE_TABLE[f as usize]
}

impl Feature {
    pub fn name(&self) -> &'static str {
        feature_info(*self).name
    }

    pub fn level(&self) -> IsaLevel {
        feature_info(*self).level
    }

    pub fn arch(&self) -> Arch {
        self.level().arch()
    }

    /// Looks up a feature by LLVM spelling. Also accepts the Rust spellings that differ
    /// (`sse4_1`/`sse41`, `bmi1`, `cmpxchg16b`, `fp16`, `paca`) so either convention works.
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.trim().to_ascii_lowercase();
        let canonical = match n.as_str() {
            "sse41" | "sse4_1" => "sse4.1",
            "sse42" | "sse4_2" => "sse4.2",
            "bmi1" => "bmi",
            "cmpxchg16b" => "cx16",
            "fp16" => "fullfp16",
            "paca" => "pauth",
            "amx_tile" => "amx-tile",
            "amx_bf16" => "amx-bf16",
            "amx_int8" => "amx-int8",
            "amx_fp16" => "amx-fp16",
            other => other,
        };
        FEATURE_TABLE
            .iter()
            .find(|i| i.name == canonical)
            .map(|i| i.feature)
    }
}

/// Representative feature that makes a level "available".
fn level_marker(level: IsaLevel) -> Option<Feature> {
    match level {
        // x86-64 baseline (SSE2) always qualifies for the lowest tier.
        IsaLevel::Sse => None,
        IsaLevel::Avx => Some(F::Avx),
        IsaLevel::Avx2 => Some(F::Avx2),
        IsaLevel::Avx512 => Some(F::Avx512f),
        IsaLevel::Amx => Some(F::AmxTile),
        IsaLevel::Neon => None,
        IsaLevel::Sve => Some(F::Sve),
        IsaLevel::Sve2 => Some(F::Sve2),
        IsaLevel::Sme => Some(F::Sme),
    }
}

/// Features that `f` requires directly (LLVM's implication rules for the features we track).
fn prerequisites(f: Feature) -> &'static [Feature] {
    match f {
        F::Ssse3 => &[F::Sse3],
        F::Sse41 => &[F::Ssse3],
        F::Sse42 => &[F::Sse41],
        F::Avx => &[F::Sse42],
        F::F16c | F::Avx2 | F::Fma => &[F::Avx],
        F::AvxVnni => &[F::Avx2],
        F::Avx512f => &[F::Avx2, F::Fma, F::F16c],
        F::Avx512vl | F::Avx512bw | F::Avx512dq | F::Avx512cd | F::Avx512vnni => &[F::Avx512f],
        F::Avx512bf16 | F::Avx512bitalg | F::Avx512vbmi => &[F::Avx512bw],
        F::Avx512fp16 => &[F::Avx512bw, F::Avx512vl, F::Avx512dq],
        F::AmxBf16 | F::AmxInt8 | F::AmxFp16 => &[F::AmxTile],
        F::Fp16 | F::Dotprod | F::I8mm | F::Sve => &[F::Neon],
        F::Sve2 => &[F::Sve],
        F::Sme2 => &[F::Sme],
        _ => &[],
    }
}

/// True if `f` requires `dep`, directly or transitively.
fn requires(f: Feature, dep: Feature) -> bool {
    prerequisites(f)
        .iter()
        .any(|&p| p == dep || requires(p, dep))
}

/// A set of CPU features plus scalable vector lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    pub arch: Arch,
    bits: u64,
    /// SVE vector length in bits (non-streaming mode), when SVE is present and known.
    pub sve_vector_bits: Option<u32>,
    /// SME streaming vector length in bits, when SME is present and known.
    pub sme_vector_bits: Option<u32>,
}

impl CpuFeatures {
    pub fn empty(arch: Arch) -> Self {
        Self {
            arch,
            bits: 0,
            sve_vector_bits: None,
            sme_vector_bits: None,
        }
    }

    /// Builds a feature set from explicit features (used for targets and tests).
    pub fn from_features(arch: Arch, features: &[Feature]) -> Self {
        let mut set = Self::empty(arch);
        for &f in features {
            set.insert(f);
        }
        set
    }

    /// Detected host features, cached for the life of the process.
    pub fn host() -> Self {
        static HOST: OnceLock<CpuFeatures> = OnceLock::new();
        *HOST.get_or_init(detect::host_features)
    }

    /// Host features with the active ISA cap applied.
    pub fn effective() -> Result<Self> {
        let host = Self::host();
        Ok(match isa_cap()? {
            Some(cap) => host.capped(cap)?,
            None => host,
        })
    }

    pub fn has(&self, f: Feature) -> bool {
        self.bits & (1u64 << f as u8) != 0
    }

    pub fn insert(&mut self, f: Feature) {
        self.bits |= 1u64 << f as u8;
    }

    pub fn remove(&mut self, f: Feature) {
        self.bits &= !(1u64 << f as u8);
    }

    /// Enables `f` and everything it requires (LLVM `+feature` semantics).
    pub fn enable(&mut self, f: Feature) {
        self.insert(f);
        for &p in prerequisites(f) {
            self.enable(p);
        }
    }

    /// Disables `f` and every feature that requires it (LLVM `-feature` semantics).
    pub fn disable(&mut self, f: Feature) {
        self.remove(f);
        for info in FEATURE_TABLE {
            if self.has(info.feature) && requires(info.feature, f) {
                self.remove(info.feature);
            }
        }
    }

    /// Adds every prerequisite of the enabled features.
    pub fn with_prerequisites(mut self) -> Self {
        for f in self.iter().collect::<Vec<_>>() {
            self.enable(f);
        }
        self
    }

    pub fn is_subset_of(&self, other: &CpuFeatures) -> bool {
        self.arch == other.arch && self.bits & !other.bits == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = Feature> + '_ {
        FEATURE_TABLE
            .iter()
            .map(|i| i.feature)
            .filter(|&f| self.has(f))
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.iter().map(|f| f.name()).collect()
    }

    /// Drops every feature above `cap`. Errors if `cap` belongs to another architecture.
    pub fn capped(&self, cap: IsaLevel) -> Result<Self> {
        if cap.arch() != self.arch {
            return Err(anyhow!(
                "[ERR_ISA_ARCH_MISMATCH] ISA level '{cap}' is for {}, but the target is {}",
                cap.arch().as_str(),
                self.arch.as_str()
            ));
        }
        let mut out = *self;
        for f in self.iter() {
            if f.level().rank() > cap.rank() {
                out.remove(f);
            }
        }
        if !out.has(F::Sve) {
            out.sve_vector_bits = None;
        }
        if !out.has(F::Sme) {
            out.sme_vector_bits = None;
        }
        Ok(out)
    }

    /// Highest ISA level this feature set fully reaches.
    pub fn max_level(&self) -> Option<IsaLevel> {
        ALL_LEVELS
            .iter()
            .copied()
            .filter(|l| l.arch() == self.arch)
            .filter(|l| level_marker(*l).is_none_or(|m| self.has(m)))
            .max_by_key(|l| l.rank())
    }

    /// Widest vector register a native code generator could use with these features.
    /// AVX alone is reported as 128 bits because 256-bit integer ops need AVX2.
    pub fn native_vector_bits(&self) -> u32 {
        match self.arch {
            Arch::X86_64 if self.has(F::Avx512f) => 512,
            Arch::X86_64 if self.has(F::Avx2) => 256,
            Arch::Aarch64 if self.has(F::Sve) => self.sve_vector_bits.unwrap_or(128),
            _ => 128,
        }
    }

    /// Applies an LLVM-style feature list (`+avx2,-avx512f`; a bare name means `+`).
    /// Enabling pulls in prerequisites; disabling also drops dependent features.
    pub fn apply_feature_string(&mut self, spec: &str) -> Result<()> {
        for (enable, f) in parse_feature_string(spec, self.arch)? {
            if enable {
                self.enable(f);
            } else {
                self.disable(f);
            }
        }
        Ok(())
    }

    /// Cranelift ISA settings implied by these features.
    pub fn cranelift_flags(&self) -> Vec<&'static str> {
        self.iter()
            .filter_map(|f| feature_info(f).cranelift_flag)
            .collect()
    }

    /// Features Cranelift cannot use (it only generates 128-bit vector code).
    pub fn cranelift_unused(&self) -> Vec<&'static str> {
        self.iter()
            .filter(|&f| feature_info(f).cranelift_flag.is_none() && f != F::Neon)
            .map(|f| f.name())
            .collect()
    }

    /// LLVM `+`/`-` feature string naming every table feature of this architecture, so
    /// LLVM enables exactly these features whatever its CPU model implies.
    pub fn llvm_feature_string(&self) -> String {
        FEATURE_TABLE
            .iter()
            .filter(|i| i.level.arch() == self.arch)
            .map(|i| {
                let sign = if self.has(i.feature) { '+' } else { '-' };
                format!("{sign}{}", i.name)
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Shape of `vx` and the preferred vector register width for LLVM code using these
    /// features: 512/256/128-bit fixed vectors on x86 by AVX-512F/AVX2, scalable vectors on
    /// SVE (with the exact vscale for JIT code, which runs on this host), else 128 bits.
    #[cfg(feature = "llvm")]
    pub fn llvm_vector_shape(&self, jit: bool) -> (achainsaw_llvm::VxShape, Option<u32>) {
        use achainsaw_llvm::VxShape;
        match self.arch {
            Arch::Aarch64 if self.has(F::Sve) => {
                let vscale_range = match (jit, self.sve_vector_bits) {
                    (true, Some(bits)) if bits >= 128 => (bits / 128, bits / 128),
                    _ => (1, 16),
                };
                (VxShape::Scalable { vscale_range }, None)
            }
            _ => {
                let bits = self.native_vector_bits();
                (VxShape::Fixed(bits), (bits > 128).then_some(bits))
            }
        }
    }

    /// Matrix engines LLVM `mm` kernels may use with these features.
    #[cfg(feature = "llvm")]
    pub fn llvm_matrix_units(&self) -> achainsaw_llvm::MatrixUnits {
        let amx = self.arch == Arch::X86_64 && self.has(F::AmxTile);
        achainsaw_llvm::MatrixUnits {
            amx_bf16: amx && self.has(F::AmxBf16),
            amx_int8: amx && self.has(F::AmxInt8),
            amx_fp16: amx && self.has(F::AmxFp16),
            sme: self.arch == Arch::Aarch64 && self.has(F::Sme),
        }
    }

    /// Whether LLVM code for these features uses scalable (SVE) `vx` vectors.
    pub fn llvm_vx_scalable(&self) -> bool {
        self.arch == Arch::Aarch64 && self.has(F::Sve)
    }

    /// LLVM CPU name and feature string for JIT code using exactly these features. The
    /// host's CPU model (for scheduling) is used only when nothing is capped, so a capped
    /// tier cannot pick up model-implied features outside the feature table.
    #[cfg(feature = "llvm")]
    pub fn llvm_target(&self) -> (String, String) {
        let cpu = match self.arch {
            _ if *self == Self::host() => achainsaw_llvm::host_cpu_name(),
            Arch::X86_64 => "x86-64".to_string(),
            Arch::Aarch64 => "generic".to_string(),
            Arch::Other => achainsaw_llvm::host_cpu_name(),
        };
        (cpu, self.llvm_feature_string())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "arch": self.arch.as_str(),
            "features": self.names(),
            "max_isa": self.max_level().map(|l| l.as_str()),
            "native_vector_bits": self.native_vector_bits(),
            "sve_vector_bits": self.sve_vector_bits,
            "sme_vector_bits": self.sme_vector_bits,
        })
    }
}

/// Parses an LLVM-style feature list into `(enable, feature)` pairs for `arch`.
pub fn parse_feature_string(spec: &str, arch: Arch) -> Result<Vec<(bool, Feature)>> {
    let mut out = Vec::new();
    for raw in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (enable, name) = match raw.as_bytes()[0] {
            b'+' => (true, &raw[1..]),
            b'-' => (false, &raw[1..]),
            _ => (true, raw),
        };
        let f = Feature::from_name(name).ok_or_else(|| {
            anyhow!(
                "[ERR_UNKNOWN_TARGET_FEATURE] Unknown target feature '{name}'; known features: {}",
                FEATURE_TABLE
                    .iter()
                    .map(|i| i.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        if f.arch() != arch {
            return Err(anyhow!(
                "[ERR_ISA_ARCH_MISMATCH] Target feature '{name}' is for {}, but the target is {}",
                f.arch().as_str(),
                arch.as_str()
            ));
        }
        out.push((enable, f));
    }
    Ok(out)
}

/// Full target report used by `achainsaw cpu`, the MCP `air_target` tool, and Python.
pub fn target_report() -> Result<Value> {
    let host = CpuFeatures::host();
    let effective = CpuFeatures::effective()?;
    Ok(json!({
        "status": "ok",
        "host": host.to_json(),
        "isa_cap": isa_cap()?.map(|l| l.as_str()),
        "effective": effective.to_json(),
        "backends": {
            "cranelift": {
                "available": true,
                "vector_bits": CRANELIFT_VECTOR_BITS,
                "unused_features": effective.cranelift_unused(),
            },
            "llvm": if cfg!(feature = "llvm") {
                json!({
                    "available": true,
                    "vector_bits": effective.native_vector_bits(),
                    "vx_scalable": effective.llvm_vx_scalable(),
                })
            } else {
                json!({ "available": false })
            },
        },
    }))
}

// ---------------------------------------------------------------------------------------------
// ISA cap
// ---------------------------------------------------------------------------------------------

/// `None` = not yet initialized from the environment.
static ISA_CAP: RwLock<Option<Option<IsaLevel>>> = RwLock::new(None);

/// Active ISA cap. Initialized from `ACHAINSAW_MAX_ISA` on first use.
pub fn isa_cap() -> Result<Option<IsaLevel>> {
    if let Some(cap) = *ISA_CAP.read().unwrap() {
        return Ok(cap);
    }
    let from_env = match std::env::var(ISA_CAP_ENV) {
        Ok(v) if !v.trim().is_empty() => Some(
            v.parse::<IsaLevel>()
                .map_err(|e| anyhow!("[ERR_INVALID_ISA_LEVEL] {ISA_CAP_ENV}: {e}"))?,
        ),
        _ => None,
    };
    if let Some(cap) = from_env {
        check_cap_arch(cap)?;
    }
    let mut guard = ISA_CAP.write().unwrap();
    Ok(*guard.get_or_insert(from_env))
}

/// Overrides the ISA cap for this process (`None` removes it, ignoring the environment).
pub fn set_isa_cap(cap: Option<IsaLevel>) -> Result<()> {
    if let Some(c) = cap {
        check_cap_arch(c)?;
    }
    *ISA_CAP.write().unwrap() = Some(cap);
    Ok(())
}

fn check_cap_arch(cap: IsaLevel) -> Result<()> {
    let host = Arch::host();
    if cap.arch() != host {
        return Err(anyhow!(
            "[ERR_ISA_ARCH_MISMATCH] ISA level '{cap}' is for {}, but this host is {}",
            cap.arch().as_str(),
            host.as_str()
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Cranelift ISA construction
// ---------------------------------------------------------------------------------------------

/// Cranelift ISA builder for the host, honoring the ISA cap.
pub fn host_isa_builder() -> Result<isa::Builder> {
    native_isa_builder(&CpuFeatures::effective()?)
}

/// Cranelift ISA builder for this host restricted to `effective` (a subset of host features).
pub fn native_isa_builder(effective: &CpuFeatures) -> Result<isa::Builder> {
    if effective.arch != Arch::host() {
        return Err(anyhow!(
            "[ERR_ISA_ARCH_MISMATCH] Features are for {}, but this host is {}",
            effective.arch.as_str(),
            Arch::host().as_str()
        ));
    }
    let host = CpuFeatures::host();
    if !effective.is_subset_of(&host) {
        let missing: Vec<&str> = effective
            .iter()
            .filter(|&f| !host.has(f))
            .map(|f| f.name())
            .collect();
        return Err(anyhow!(
            "[ERR_UNSUPPORTED_FEATURE] This CPU lacks {}; JIT code using them would fault",
            missing.join(", ")
        ));
    }
    if effective.arch == Arch::Other {
        // No feature table for this architecture; let Cranelift infer host flags itself.
        return cranelift_native::builder()
            .map_err(|msg| anyhow!("Host machine not supported by Cranelift: {msg}"));
    }
    let mut builder = cranelift_native::builder_with_options(false)
        .map_err(|msg| anyhow!("Host machine not supported by Cranelift: {msg}"))?;
    apply_cranelift_flags(&mut builder, effective)?;
    Ok(builder)
}

/// Cranelift ISA builder for an AOT target.
///
/// - No triple and no CPU: the capped host features, same as the JIT.
/// - `cpu`: a Cranelift preset instead of host detection. x86 presets use LLVM CPU names
///   (`znver4`, `sapphirerapids`, `x86-64-v3`); aarch64 only accepts `generic`.
/// - `features`: LLVM-style overrides (`+avx2,-avx512f`) applied last.
///
/// Returns the builder plus the requested features Cranelift cannot use, so callers can
/// report that they had no effect on the generated code.
pub fn target_isa_builder(
    triple: Option<&str>,
    cpu: Option<&str>,
    features: Option<&str>,
) -> Result<(isa::Builder, Vec<&'static str>)> {
    let (mut builder, arch) = match triple {
        Some(t) => {
            let triple = target_lexicon::Triple::from_str(t)
                .map_err(|e| anyhow!("[ERR_INVALID_TARGET] Invalid target triple '{t}': {e}"))?;
            let arch = match triple.architecture {
                target_lexicon::Architecture::X86_64 => Arch::X86_64,
                target_lexicon::Architecture::Aarch64(_) => Arch::Aarch64,
                _ => Arch::Other,
            };
            let builder = isa::lookup(triple).map_err(|e| {
                anyhow!(
                    "[ERR_UNSUPPORTED_TARGET] No Cranelift backend for '{t}' in this build: {e}"
                )
            })?;
            (builder, arch)
        }
        None if cpu.is_some() => (
            cranelift_native::builder_with_options(false)
                .map_err(|msg| anyhow!("Host machine not supported by Cranelift: {msg}"))?,
            Arch::host(),
        ),
        None => (host_isa_builder()?, Arch::host()),
    };

    let explicit = triple.is_some() || cpu.is_some();
    if let Some(cpu) = cpu {
        let ok = match arch {
            Arch::X86_64 => builder.enable(cpu).is_ok(),
            _ => cpu == "generic",
        };
        if !ok {
            return Err(anyhow!(
                "[ERR_UNKNOWN_TARGET_CPU] Unknown target CPU '{cpu}' for {}; x86_64 accepts LLVM CPU names such as x86-64-v3, znver4, sapphirerapids",
                arch.as_str()
            ));
        }
    }
    if arch == Arch::Other {
        if features.is_some() {
            return Err(anyhow!(
                "[ERR_UNKNOWN_TARGET_FEATURE] Target features are only supported for x86_64 and aarch64"
            ));
        }
        return Ok((builder, Vec::new()));
    }

    // Start from what the target already enables, then complete prerequisites: Cranelift's
    // generic x86-64-v3/v4 presets omit `has_avx` (which gates all VEX encodings) and v4 omits
    // `has_avx512f`, so taking presets at face value silently produces SSE-only code.
    let mut set = if explicit {
        enabled_cranelift_features(&builder, arch)?.with_prerequisites()
    } else {
        CpuFeatures::effective()?
    };

    let mut ignored = Vec::new();
    if let Some(spec) = features {
        for (enable, f) in parse_feature_string(spec, arch)? {
            if enable {
                set.enable(f);
                if feature_info(f).cranelift_flag.is_none() && f != F::Neon {
                    ignored.push(f.name());
                }
            } else {
                set.disable(f);
            }
        }
    }
    apply_cranelift_flags(&mut builder, &set)?;
    Ok((builder, ignored))
}

/// LLVM target for an AOT build, with the same defaults as [`target_isa_builder`]: the
/// capped host when neither a triple nor a CPU is given, otherwise the named CPU (or the
/// architecture's generic model) plus `features` overrides, validated against the table.
#[cfg(feature = "llvm")]
pub fn llvm_aot_target(
    target: &crate::aot::AotTarget,
) -> Result<(achainsaw_llvm::TargetSpec, CpuFeatures)> {
    let arch = match target.triple.as_deref() {
        None => Arch::host(),
        Some(t) => {
            let triple = target_lexicon::Triple::from_str(t)
                .map_err(|e| anyhow!("[ERR_INVALID_TARGET] Invalid target triple '{t}': {e}"))?;
            match triple.architecture {
                target_lexicon::Architecture::X86_64 => Arch::X86_64,
                target_lexicon::Architecture::Aarch64(_) => Arch::Aarch64,
                _ => Arch::Other,
            }
        }
    };
    let overrides = match target.features.as_deref() {
        Some(spec) if arch == Arch::Other => {
            return Err(anyhow!(
                "[ERR_UNKNOWN_TARGET_FEATURE] Target features are only supported for x86_64 and aarch64 (got '{spec}')"
            ))
        }
        Some(spec) => {
            parse_feature_string(spec, arch)?;
            spec.to_string()
        }
        None => String::new(),
    };
    if let (Arch::X86_64, Some(cpu)) = (arch, &target.cpu) {
        // LLVM only warns about unknown CPUs and falls back to a generic model. x86 CPU
        // presets use LLVM names, so reject unknown ones with the Cranelift error instead.
        let x86 = target_lexicon::Triple::from_str("x86_64-unknown-linux-gnu").unwrap();
        if let Ok(mut builder) = isa::lookup(x86) {
            if builder.enable(cpu).is_err() {
                return Err(anyhow!(
                    "[ERR_UNKNOWN_TARGET_CPU] Unknown target CPU '{cpu}' for x86_64; use LLVM CPU names such as x86-64-v3, znver4, sapphirerapids"
                ));
            }
        }
    }
    let (cpu, base) = match (&target.triple, &target.cpu) {
        (None, None) => CpuFeatures::effective()?.llvm_target(),
        (_, Some(cpu)) => (cpu.clone(), String::new()),
        (Some(_), None) => (
            match arch {
                Arch::X86_64 => "x86-64",
                _ => "generic",
            }
            .to_string(),
            String::new(),
        ),
    };
    // Our view of the target's features, used to shape `vx`: the capped host, or the CPU
    // preset (x86 only; other named CPUs contribute nothing), plus the overrides.
    let mut set = match (&target.triple, &target.cpu, arch) {
        (None, None, _) => CpuFeatures::effective()?,
        (_, Some(cpu), Arch::X86_64) => {
            let x86 = target_lexicon::Triple::from_str("x86_64-unknown-linux-gnu").unwrap();
            match isa::lookup(x86) {
                Ok(mut builder) => {
                    let _ = builder.enable(cpu);
                    enabled_cranelift_features(&builder, arch)?.with_prerequisites()
                }
                Err(_) => CpuFeatures::empty(arch),
            }
        }
        (_, _, Arch::Aarch64) => CpuFeatures::from_features(arch, &[F::Neon]),
        _ => CpuFeatures::empty(arch),
    };
    if let Some(cpu) = &target.cpu {
        for &f in llvm_cpu_extra_features(arch, cpu) {
            set.enable(f);
        }
    }
    if !overrides.is_empty() {
        set.apply_feature_string(&overrides)?;
    }
    let features = [base, overrides]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(",");
    Ok((
        achainsaw_llvm::TargetSpec {
            triple: target.triple.clone(),
            cpu,
            features,
        },
        set,
    ))
}

/// Features of named CPUs that Cranelift's presets do not model but LLVM lowering depends
/// on: AMX (which `mm` kernel to use) and SVE/SME (`vx` shape and `mm` kernel).
#[cfg(feature = "llvm")]
fn llvm_cpu_extra_features(arch: Arch, cpu: &str) -> &'static [Feature] {
    match (arch, cpu.to_ascii_lowercase().as_str()) {
        (Arch::X86_64, "sapphirerapids" | "emeraldrapids") => &[F::AmxTile, F::AmxBf16, F::AmxInt8],
        (Arch::X86_64, "graniterapids" | "graniterapids-d" | "diamondrapids") => {
            &[F::AmxTile, F::AmxBf16, F::AmxInt8, F::AmxFp16]
        }
        (Arch::Aarch64, "a64fx" | "neoverse-v1") => &[F::Sve],
        (Arch::Aarch64, "neoverse-n2" | "neoverse-n3" | "neoverse-v2" | "neoverse-v3") => {
            &[F::Sve, F::Sve2]
        }
        // Apple M4 has SME (streaming mode only) but no non-streaming SVE.
        (Arch::Aarch64, "apple-m4") => &[F::Sme],
        _ => &[],
    }
}

/// Features whose Cranelift settings are currently enabled on `builder`.
fn enabled_cranelift_features(builder: &isa::Builder, arch: Arch) -> Result<CpuFeatures> {
    let isa = builder
        .finish(cranelift_codegen::settings::Flags::new(
            cranelift_codegen::settings::builder(),
        ))
        .map_err(|e| anyhow!("Invalid Cranelift ISA configuration: {e}"))?;
    let on: Vec<&str> = isa
        .isa_flags()
        .iter()
        .filter(|v| v.as_bool() == Some(true))
        .map(|v| v.name)
        .collect();
    let mut set = CpuFeatures::empty(arch);
    for info in FEATURE_TABLE.iter().filter(|i| i.level.arch() == arch) {
        if info.cranelift_flag.is_some_and(|flag| on.contains(&flag)) {
            set.insert(info.feature);
        }
    }
    Ok(set)
}

/// Enables exactly the Cranelift settings implied by `features` on `builder`.
pub fn apply_cranelift_flags(builder: &mut isa::Builder, features: &CpuFeatures) -> Result<()> {
    let enabled = features.cranelift_flags();
    for info in FEATURE_TABLE
        .iter()
        .filter(|i| i.level.arch() == features.arch)
    {
        if let Some(flag) = info.cranelift_flag {
            let value = if enabled.contains(&flag) {
                "true"
            } else {
                "false"
            };
            builder
                .set(flag, value)
                .map_err(|e| anyhow!("Cranelift rejected setting {flag}={value}: {e}"))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------------------------

mod detect {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    pub fn host_features() -> CpuFeatures {
        let mut set = CpuFeatures::empty(Arch::X86_64);
        macro_rules! probe {
            ($($name:tt => $feat:expr),* $(,)?) => {
                $(if std::is_x86_feature_detected!($name) { set.insert($feat); })*
            };
        }
        probe! {
            "sse3" => F::Sse3, "ssse3" => F::Ssse3, "sse4.1" => F::Sse41, "sse4.2" => F::Sse42,
            "popcnt" => F::Popcnt, "cmpxchg16b" => F::Cx16, "avx" => F::Avx, "f16c" => F::F16c,
            "avx2" => F::Avx2, "fma" => F::Fma, "bmi1" => F::Bmi1, "bmi2" => F::Bmi2,
            "lzcnt" => F::Lzcnt, "avxvnni" => F::AvxVnni, "avx512f" => F::Avx512f,
            "avx512vl" => F::Avx512vl, "avx512bw" => F::Avx512bw, "avx512dq" => F::Avx512dq,
            "avx512cd" => F::Avx512cd, "avx512bf16" => F::Avx512bf16,
            "avx512fp16" => F::Avx512fp16, "avx512vnni" => F::Avx512vnni,
            "avx512bitalg" => F::Avx512bitalg, "avx512vbmi" => F::Avx512vbmi,
        }
        // AMX detection is unstable in std, so read CPUID directly.
        for f in amx::usable_features() {
            set.insert(f);
        }
        set
    }

    #[cfg(target_arch = "aarch64")]
    pub fn host_features() -> CpuFeatures {
        let mut set = CpuFeatures::empty(Arch::Aarch64);
        macro_rules! probe {
            ($($name:tt => $feat:expr),* $(,)?) => {
                $(if std::arch::is_aarch64_feature_detected!($name) { set.insert($feat); })*
            };
        }
        probe! {
            "neon" => F::Neon, "fp16" => F::Fp16, "dotprod" => F::Dotprod, "i8mm" => F::I8mm,
            "lse" => F::Lse, "paca" => F::Pauth, "sve" => F::Sve, "sve2" => F::Sve2,
        }
        // SME detection is unstable in std; query the OS directly.
        let (sme, sme2) = sme::detect();
        if sme {
            set.insert(F::Sme);
        }
        if sme2 {
            set.insert(F::Sme2);
        }
        if set.has(F::Sve) {
            set.sve_vector_bits = sme::sve_vector_bits();
        }
        if set.has(F::Sme) {
            set.sme_vector_bits = sme::sme_vector_bits();
        }
        set
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn host_features() -> CpuFeatures {
        CpuFeatures::empty(Arch::Other)
    }

    #[cfg(target_arch = "x86_64")]
    mod amx {
        use super::Feature;
        use super::F;
        use std::arch::x86_64::{__cpuid_count, _xgetbv, CpuidResult};

        const XTILECFG: u64 = 1 << 17;
        const XTILEDATA: u64 = 1 << 18;

        #[target_feature(enable = "xsave")]
        unsafe fn xcr0() -> u64 {
            _xgetbv(0)
        }

        /// AMX features that the CPU has, the OS has enabled in XCR0, and (on Linux) the
        /// process has been granted permission to use.
        pub fn usable_features() -> Vec<Feature> {
            let leaf0: CpuidResult = __cpuid_count(0, 0);
            if leaf0.eax < 7 {
                return Vec::new();
            }
            let osxsave = __cpuid_count(1, 0).ecx & (1 << 27) != 0;
            let l7 = __cpuid_count(7, 0);
            let tile = l7.edx & (1 << 24) != 0;
            if !tile || !osxsave {
                return Vec::new();
            }
            // SAFETY: OSXSAVE is set, so XGETBV is available.
            let xcr0 = unsafe { xcr0() };
            if xcr0 & (XTILECFG | XTILEDATA) != (XTILECFG | XTILEDATA) {
                return Vec::new();
            }
            if !request_permission() {
                return Vec::new();
            }
            let mut out = vec![F::AmxTile];
            if l7.edx & (1 << 22) != 0 {
                out.push(F::AmxBf16);
            }
            if l7.edx & (1 << 25) != 0 {
                out.push(F::AmxInt8);
            }
            if l7.eax >= 1 && __cpuid_count(7, 1).eax & (1 << 21) != 0 {
                out.push(F::AmxFp16);
            }
            out
        }

        /// Linux keeps AMX tile data disabled until the process asks for it; without this,
        /// the first tile instruction raises SIGILL.
        #[cfg(target_os = "linux")]
        fn request_permission() -> bool {
            const ARCH_REQ_XCOMP_PERM: libc::c_long = 0x1023;
            const XFEATURE_XTILEDATA: libc::c_long = 18;
            // SAFETY: arch_prctl with these arguments only changes this process's permissions.
            unsafe {
                libc::syscall(
                    libc::SYS_arch_prctl,
                    ARCH_REQ_XCOMP_PERM,
                    XFEATURE_XTILEDATA,
                ) == 0
            }
        }

        /// Other OSes (Windows) manage AMX state on demand once XCR0 enables it.
        #[cfg(not(target_os = "linux"))]
        fn request_permission() -> bool {
            true
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod sme {
        #[cfg(target_os = "linux")]
        pub fn detect() -> (bool, bool) {
            const HWCAP2_SME: u64 = 1 << 23;
            const HWCAP2_SME2: u64 = 1 << 37;
            // SAFETY: getauxval has no preconditions.
            let hwcap2 = unsafe { libc::getauxval(libc::AT_HWCAP2) } as u64;
            (hwcap2 & HWCAP2_SME != 0, hwcap2 & HWCAP2_SME2 != 0)
        }

        #[cfg(target_os = "macos")]
        pub fn detect() -> (bool, bool) {
            (
                sysctl_flag(c"hw.optional.arm.FEAT_SME"),
                sysctl_flag(c"hw.optional.arm.FEAT_SME2"),
            )
        }

        #[cfg(target_os = "macos")]
        fn sysctl_flag(name: &std::ffi::CStr) -> bool {
            let mut value: libc::c_int = 0;
            let mut size = std::mem::size_of::<libc::c_int>();
            // SAFETY: value/size describe a valid c_int buffer.
            let rc = unsafe {
                libc::sysctlbyname(
                    name.as_ptr(),
                    (&mut value as *mut libc::c_int).cast(),
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };
            rc == 0 && value != 0
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        pub fn detect() -> (bool, bool) {
            (false, false)
        }

        #[cfg(target_os = "linux")]
        fn prctl_vector_bits(option: libc::c_int) -> Option<u32> {
            const VL_LEN_MASK: libc::c_int = 0xffff;
            // SAFETY: PR_SVE_GET_VL / PR_SME_GET_VL take no further arguments.
            let rc = unsafe { libc::prctl(option) };
            (rc > 0).then(|| ((rc & VL_LEN_MASK) as u32) * 8)
        }

        #[cfg(target_os = "linux")]
        pub fn sve_vector_bits() -> Option<u32> {
            const PR_SVE_GET_VL: libc::c_int = 51;
            prctl_vector_bits(PR_SVE_GET_VL)
        }

        #[cfg(target_os = "linux")]
        pub fn sme_vector_bits() -> Option<u32> {
            const PR_SME_GET_VL: libc::c_int = 64;
            prctl_vector_bits(PR_SME_GET_VL)
        }

        // Vector lengths are not exposed by the OS elsewhere; reading them needs RDVL/RDSVL,
        // which arrives with the LLVM tier.
        #[cfg(not(target_os = "linux"))]
        pub fn sve_vector_bits() -> Option<u32> {
            None
        }

        #[cfg(not(target_os = "linux"))]
        pub fn sme_vector_bits() -> Option<u32> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_table_matches_enum_order() {
        for (i, info) in FEATURE_TABLE.iter().enumerate() {
            assert_eq!(info.feature as usize, i, "{} out of order", info.name);
        }
        assert!(FEATURE_TABLE.len() <= 64);
    }

    #[test]
    fn feature_names_round_trip() {
        for info in FEATURE_TABLE {
            assert_eq!(Feature::from_name(info.name), Some(info.feature));
        }
        assert_eq!(Feature::from_name("sse4_1"), Some(F::Sse41));
        assert_eq!(Feature::from_name("bmi1"), Some(F::Bmi1));
        assert_eq!(Feature::from_name("AVX512F"), Some(F::Avx512f));
        assert_eq!(Feature::from_name("avx1024"), None);
    }

    #[test]
    fn isa_level_parsing() {
        for name in IsaLevel::names() {
            assert_eq!(name.parse::<IsaLevel>().unwrap().as_str(), name);
        }
        assert_eq!(" AVX2 ".parse::<IsaLevel>(), Ok(IsaLevel::Avx2));
        assert!("avx3".parse::<IsaLevel>().unwrap_err().contains("avx512"));
    }

    fn zen4_like() -> CpuFeatures {
        CpuFeatures::from_features(
            Arch::X86_64,
            &[
                F::Sse3,
                F::Ssse3,
                F::Sse41,
                F::Sse42,
                F::Popcnt,
                F::Cx16,
                F::Avx,
                F::F16c,
                F::Avx2,
                F::Fma,
                F::Bmi1,
                F::Bmi2,
                F::Lzcnt,
                F::Avx512f,
                F::Avx512vl,
                F::Avx512bw,
                F::Avx512dq,
                F::Avx512bf16,
                F::Avx512vnni,
            ],
        )
    }

    #[test]
    fn capping_drops_higher_tiers() {
        let full = zen4_like();
        assert_eq!(full.max_level(), Some(IsaLevel::Avx512));
        assert_eq!(full.native_vector_bits(), 512);

        let avx2 = full.capped(IsaLevel::Avx2).unwrap();
        assert!(avx2.has(F::Fma) && !avx2.has(F::Avx512f) && !avx2.has(F::Avx512bf16));
        assert_eq!(avx2.max_level(), Some(IsaLevel::Avx2));
        assert_eq!(avx2.native_vector_bits(), 256);

        let avx = full.capped(IsaLevel::Avx).unwrap();
        assert!(avx.has(F::Avx) && !avx.has(F::Avx2));
        assert_eq!(avx.native_vector_bits(), 128);

        let sse = full.capped(IsaLevel::Sse).unwrap();
        assert_eq!(sse.max_level(), Some(IsaLevel::Sse));
        assert!(sse.has(F::Sse42) && !sse.has(F::Avx));

        assert!(full.capped(IsaLevel::Sve).is_err());
    }

    #[test]
    fn sve_vector_length_follows_cap() {
        let mut arm =
            CpuFeatures::from_features(Arch::Aarch64, &[F::Neon, F::Lse, F::Sve, F::Sve2, F::Sme]);
        arm.sve_vector_bits = Some(256);
        arm.sme_vector_bits = Some(512);
        assert_eq!(arm.max_level(), Some(IsaLevel::Sme));
        assert_eq!(arm.native_vector_bits(), 256);

        let neon = arm.capped(IsaLevel::Neon).unwrap();
        assert_eq!(neon.native_vector_bits(), 128);
        assert_eq!(neon.sve_vector_bits, None);
        assert_eq!(neon.sme_vector_bits, None);
        assert!(neon.has(F::Lse));
    }

    #[test]
    fn feature_strings() {
        let mut f = zen4_like();
        f.apply_feature_string("-avx512f, +amx-tile,avxvnni")
            .unwrap();
        assert!(!f.has(F::Avx512f) && f.has(F::AmxTile) && f.has(F::AvxVnni));
        let err = f.apply_feature_string("+sve").unwrap_err().to_string();
        assert!(err.contains("ERR_ISA_ARCH_MISMATCH"), "{err}");
        let err = f.apply_feature_string("+avx9").unwrap_err().to_string();
        assert!(err.contains("ERR_UNKNOWN_TARGET_FEATURE"), "{err}");
    }

    #[test]
    fn enable_and_disable_follow_dependencies() {
        let mut f = CpuFeatures::empty(Arch::X86_64);
        f.enable(F::Avx512vl);
        for dep in [F::Avx512f, F::Avx2, F::Fma, F::Avx, F::Sse42, F::Sse3] {
            assert!(f.has(dep), "{} should be implied", dep.name());
        }
        f.disable(F::Avx);
        assert!(f.has(F::Sse42));
        assert!(!f.has(F::Avx2) && !f.has(F::Avx512f) && !f.has(F::Avx512vl));

        let mut arm = CpuFeatures::empty(Arch::Aarch64);
        arm.enable(F::Sve2);
        assert!(arm.has(F::Sve) && arm.has(F::Neon));
        arm.enable(F::Sme2);
        assert!(arm.has(F::Sme));
        arm.disable(F::Sve);
        assert!(!arm.has(F::Sve2) && arm.has(F::Sme2));
    }

    #[test]
    fn generic_x86_presets_get_avx() {
        for (cpu, expected) in [("x86-64-v3", F::Avx), ("x86-64-v4", F::Avx512f)] {
            let (builder, _) =
                target_isa_builder(Some("x86_64-unknown-linux-gnu"), Some(cpu), None).unwrap();
            let set = enabled_cranelift_features(&builder, Arch::X86_64).unwrap();
            assert!(set.has(expected), "{cpu} should enable {}", expected.name());
        }
    }

    #[test]
    fn cranelift_flags_cover_vector_features() {
        let f = zen4_like();
        let flags = f.cranelift_flags();
        assert!(flags.contains(&"has_avx2") && flags.contains(&"has_avx512f"));
        let unused = f.cranelift_unused();
        assert!(unused.contains(&"avx512bw") && unused.contains(&"avx512bf16"));
        assert!(!unused.contains(&"avx2"));
    }

    #[test]
    fn every_cranelift_flag_is_accepted_for_its_arch() {
        for (triple, arch) in [
            ("x86_64-unknown-linux-gnu", Arch::X86_64),
            ("aarch64-unknown-linux-gnu", Arch::Aarch64),
        ] {
            let all: Vec<Feature> = FEATURE_TABLE
                .iter()
                .filter(|i| i.level.arch() == arch)
                .map(|i| i.feature)
                .collect();
            let feats = CpuFeatures::from_features(arch, &all);
            let mut builder = isa::lookup_by_name(triple).unwrap();
            apply_cranelift_flags(&mut builder, &feats).unwrap();
        }
    }

    #[test]
    fn jit_isa_rejects_features_the_host_lacks() {
        let host = CpuFeatures::host();
        assert!(native_isa_builder(&host).is_ok());
        let missing = FEATURE_TABLE
            .iter()
            .map(|i| i.feature)
            .find(|&f| f.arch() == host.arch && !host.has(f));
        if let Some(f) = missing {
            let mut more = host;
            more.insert(f);
            let err = native_isa_builder(&more).err().unwrap().to_string();
            assert!(
                err.contains("ERR_UNSUPPORTED_FEATURE") && err.contains(f.name()),
                "{err}"
            );
        }
    }

    #[test]
    fn host_detection_is_consistent() {
        let host = CpuFeatures::host();
        assert_eq!(host.arch, Arch::host());
        // Feature-detection macros only exist on their own architecture.
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(host.has(F::Avx2), std::is_x86_feature_detected!("avx2"));
            assert_eq!(
                host.has(F::Avx512f),
                std::is_x86_feature_detected!("avx512f")
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(host.has(F::Neon));
            assert_eq!(
                host.has(F::Sve),
                std::arch::is_aarch64_feature_detected!("sve")
            );
            assert_eq!(
                host.sve_vector_bits.is_some(),
                cfg!(target_os = "linux") && host.has(F::Sve)
            );
        }
        let report = target_report().unwrap();
        assert_eq!(
            report["backends"]["cranelift"]["vector_bits"],
            CRANELIFT_VECTOR_BITS
        );
    }
}

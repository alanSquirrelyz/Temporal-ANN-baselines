# Synthetic Timestamps for Range-Filtered ANN Benchmarks

> Companion documentation for `synth_timestamps.py`, a single-file generator
> that augments vector datasets (`.fbin` / `.u8bin` / `.i8bin`) with synthetic
> per-point timestamps and matched range-filtered query workloads, byte-
> compatible with the ANNlib artifact format
> (`ANNlib/test/generate_dataset/point_timerange_ann/timerange_dataset.hpp`)
> and consumable by the Python wrappers in
> `scripts/temporal_ann_baselines/lib/common.py`.

---

## Table of contents

1.  [Scope and contribution](#1-scope-and-contribution)
2.  [Notation and conventions](#2-notation-and-conventions)
3.  [Artifact format (formal spec)](#3-artifact-format-formal-spec)
4.  [Rank-compression invariance — statement and consequence](#4-rank-compression-invariance--statement-and-consequence)
5.  [Distribution catalogue](#5-distribution-catalogue)
    - 5.1 Family overview
    - 5.2 `uniform`
    - 5.3 `pareto_recent`
    - 5.4 `lognormal_recent`
    - 5.5 `topic_drift`
    - 5.6 `topic_burst`
    - 5.7 `topic_seasonal`
6.  [Query / window workload](#6-query--window-workload)
7.  [Ground truth: brute force with windowing](#7-ground-truth-brute-force-with-windowing)
8.  [Recommended experimental matrix and statistical reporting](#8-recommended-experimental-matrix-and-statistical-reporting)
9.  [Reproducibility checklist](#9-reproducibility-checklist)
10. [Validation and sanity checks](#10-validation-and-sanity-checks)
11. [Limitations and threats to validity](#11-limitations-and-threats-to-validity)
12. [Bibliography (BibTeX)](#12-bibliography-bibtex)

---

## 1. Scope and contribution

### 1.1 Why this exists

Range-filtered ANN (RF-ANN) studies — SeRF [ZuoEtAl2024], iRangeGraph
[XuEtAl2024], DSG [DSG2025], UNIFY [UNIFY2024], Window-Filter ANN
[EngelsEtAl2024] — assume each base vector carries a scalar attribute
(often interpreted as a timestamp) and that each query carries an interval
`[lo, hi]` on that attribute. ANN-benchmark vector datasets (BIGANN,
Yandex-DEEP, OpenAI-text-embedding, Cohere) ship without natural timestamps.
ANNlib's reference generator (`timerange_dataset.hpp::generate_point_timestamps`)
assigns timestamps i.i.d. uniformly, which makes the attribute statistically
**independent of vector content**.

The discriminative power of an RF-ANN benchmark on such timestamps is
**limited** (see §4): in the standard rank-compressed evaluation that the
Python wrappers perform, all independent univariate timestamp distributions
collapse to a single equivalence class.

This module adds six timestamp distributions, **including three that couple
timestamps with vector content**, so that benchmark results can probe
behaviors not visible under the reference uniform workload.

### 1.2 What this is *not*

This is not a *temporal ANN index*; it is a *workload generator* that
produces inputs to existing RF-ANN indices. No index code is added.

It is also not a study of which distribution best models a *particular* real
corpus; rather, it provides a catalogue of mathematically grounded options
drawn from the temporal-modelling and topic-modelling literature.

### 1.3 Reproducibility

All distributions consume a single `numpy.random.Generator` seeded by
`--seed`. K-means (for `topic_*`) derives its seed from the same generator.
Every artifact directory writes a `manifest.txt` recording the exact CLI used.

---

## 2. Notation and conventions

| Symbol | Meaning |
|---|---|
| `n` | number of base vectors after `--max-points` truncation |
| `d` | vector dimensionality |
| `m` | number of queries (`--num-queries`) |
| `k` | top-k for ground truth (`--query-k`) |
| `T` | length of the discrete timeline; we set `T = n` so timestamps and ranks share a numerical range |
| `t_i` | timestamp of the i-th base vector, `t_i ∈ {0, …, T-1}` |
| `x_i` | the i-th base vector, `x_i ∈ ℝ^d` |
| `K` | number of clusters (`--num-clusters`), `topic_*` only |
| `c_i` | k-means cluster assignment of the i-th point, `c_i ∈ {1, …, K}` |
| `W` | target window width in **rank units** (number of points) |
| `[lo_q, hi_q]` | the q-th query's closed timestamp interval |
| `Q_p(·)` | empirical p-quantile |
| `Pa(α)` | Pareto distribution, tail index α |
| `LogN(μ, σ²)` | log-normal distribution |
| `N(μ, σ²)` | Gaussian distribution |
| `Beta(a, b)` | Beta distribution |

All timestamps and IDs are stored on disk as **little-endian `uint32`**.

---

## 3. Artifact format (formal spec)

All generators emit four file families into `--out-dir`. Layouts match
`ANNlib/test/generate_dataset/point_timerange_ann/timerange_dataset.hpp`
exactly (verified line-by-line below).

### 3.1 `point_timestamps.bin`

```
struct PointTimestampsFile {
    uint32_t n;             // base vector count
    uint32_t ts[n];         // per-point timestamps, row-aligned to the base file
};
```

Reader: `lib.common.read_u32_array` at `lib/common.py:129`.
Writer (reference): `timerange_dataset.hpp:85-95`.

### 3.2 `query_ids.bin`

```
struct QueryIdsFile {
    uint32_t m;             // query count
    uint32_t qid[m];        // base-vector IDs used as queries
};
```

The query vectors themselves are `base[qid]` (verified at `lib/common.py:260`).
This convention matches both ANNlib and SeRF's `data_wrapper`.

### 3.3 `range<W>.query_ranges.bin`

```
struct QueryRangesFile {
    uint32_t m;             // query count (same as query_ids.bin)
    struct { uint32_t begin; uint32_t end; } ranges[m];   // closed interval
};
```

Reader: `lib.common.read_query_ranges` at `lib/common.py:141`.
Writer (reference): `timerange_dataset.hpp:108-117`.

The filename `range<W>` encodes the **target rank-width** (cf. §6). Multiple
widths produce multiple files, each consumed independently by the wrappers.

### 3.4 `range<W>.gt.ibin`

```
struct GroundTruthFile {
    uint32_t num_queries;
    uint32_t k;
    uint32_t id[num_queries * k];     // row-major; padding = 0xFFFFFFFF
};
```

Reader: `lib.common.read_timerange_groundtruth` at `lib/common.py:153`.
Writer (reference): `timerange_dataset.hpp:153-171`.

Padding sentinel `0xFFFFFFFF` is identical to `std::numeric_limits<uint32_t>::max()`
in ANNlib's writer (verified `timerange_dataset.hpp:164`). It also equals `-1`
when reinterpreted as `int32`, which is the convention the wrappers map it back
to (`lib/common.py:274`).

### 3.5 `manifest.txt`

Plain `key=value` lines (one per CLI argument) for reproducibility:

```
dataset=/data/zshen055/ANN/Yandex-DEEP/base.1B.fbin:fbin
out_dir=/data/yliu908/temporal-ANN/.../deep1M_yandex__topic_drift
max_points=1000000
num_queries=1000
query_k=10
ranges=100 1000 10000 100000
dist=L2
distribution=topic_drift
center_bias=uniform
seed=42
pareto_alpha=1.2
lognormal_mu=0.0
lognormal_sigma=1.0
num_clusters=128
sigma_frac=0.05
hawkes_mu=0.5
hawkes_alpha=0.6
hawkes_beta=1.0
seasonal_period_frac=0.142857
seasonal_amp=0.6
```

---

## 4. Rank-compression invariance — statement and consequence

The Python wrappers rank-compress per-point timestamps before invoking the
underlying baselines (`lib/common.py:212-240`, `compress_scalars_and_ranges`).

**Definition (rank compression).** Let `v_0 < v_1 < … < v_{U-1}` denote the
sorted unique values of `(t_1, …, t_n)`. Define `rank(t_i) := j` where
`v_j = t_i`. Map each query interval `[lo, hi]` to
`[rank_left(lo), rank_right(hi)]` via two `searchsorted` calls. Membership is
preserved: `{i : t_i ∈ [lo, hi]} = {i : rank(t_i) ∈ [rank_left(lo),
rank_right(hi)]}`.

**Proposition 4.1 (rank-compression invariance).** Let the timestamp vector
`(t_1, …, t_n)` have distribution `P` over `[0, T)^n` such that (i) the joint
distribution is exchangeable with respect to the point indices, and (ii) the
marginal CDF `F_P` is continuous. Then for any query workload
`Q = {[lo_q, hi_q]}` whose `(lo_q, hi_q)` are determined by `t_{1:n}` only
through their ranks, the joint distribution of
`(window membership at query q)_q` is identical for any two such
distributions `P, P'`.

*Proof sketch.* Under exchangeability and continuity, `rank(·)` is a uniform
random permutation of `{0, …, n-1}` independent of the labelling of points.
Window membership is a function of ranks alone (by definition of rank
compression and by the workload assumption). The distribution of ranks is the
same for any exchangeable continuous `P`; therefore window-membership
statistics are identical. ∎

**Consequence for this benchmark.** `uniform`, `pareto_recent`, and
`lognormal_recent` produce identical rank-permutation statistics, and the
downstream baselines see indistinguishable workloads in expectation. The
three are kept as **mutual controls** — agreement across them is a
sanity-check that no implementation bug breaks the invariance.

The `topic_*` distributions violate exchangeability — `c_i` (and therefore
`t_i`) depend on `x_i` — so window membership becomes correlated with vector
position. These are the distributions that meaningfully discriminate baseline
performance.

---

## 5. Distribution catalogue

### 5.1 Family overview

| Mode | Family | Coupling | Sampling cost |
|---|---|---|---|
| `uniform` | i.i.d. uniform | none | `O(n)` |
| `pareto_recent` | i.i.d. Pareto | none | `O(n)` |
| `lognormal_recent` | i.i.d. log-normal | none | `O(n)` |
| `topic_drift` | Gaussian mixture | k-means(`x`) | `O(n d / batch)` + Gaussian draws |
| `topic_burst` | Hawkes mixture (per cluster) | k-means(`x`) | `O(n)` events (Ogata thinning, O(1)/cand.) |
| `topic_seasonal` | NHPP mixture (per cluster) | k-means(`x`) | `O(n)` events (Lewis-Shedler thinning) |

**Why this set?** We chose distributions that (a) have generative paper
provenance, (b) are routinely fit to real corpora (web access, video views,
tweet bursts, social activity), (c) admit exact sampling algorithms, and (d)
together span four phenomena: **steady-state**, **recency bias**, **bursts**,
and **periodicity**. Alternatives we considered but dropped:

- **Weibull / Gamma recency.** Mathematically tidy but offer no behavior not
  already covered by Pareto/log-normal under rank compression.
- **Yule-Simon / preferential attachment.** Discrete; awkward to map to
  continuous timestamps; subsumed by Pareto in practice.
- **Fractional Brownian motion** [MandelbrotVanNess1968]. Captures
  self-similarity but is non-stationary and harder to sample at million
  scale; the per-cluster Hawkes provides similar burst structure with
  cleaner semantics.
- **Cox process** [Cox1955]. Subsumes both NHPP and Hawkes; including all
  three would inflate the matrix without adding novel baseline-side stress.

### 5.2 `uniform`

#### 5.2.1 Definition

$$P(t_i = j) = \frac{1}{T}, \quad j \in \{0, \ldots, T-1\}, \quad
t_1, \ldots, t_n \;\;\text{i.i.d.}$$

**Mean** `(T-1)/2`. **Variance** `(T² - 1)/12`. **No tail.**

#### 5.2.2 Algorithm

```
1: for i = 1 to n do
2:   t_i ← Uniform{0, …, T-1}        # numpy.Generator.integers
3: return t
```

#### 5.2.3 Complexity

Time `O(n)`. Memory `O(n)`.

#### 5.2.4 Purpose in the benchmark

Sanity baseline. Reproduces exactly ANNlib's reference workload
(`timerange_dataset.hpp:48-59`).

#### 5.2.5 Citations

- ANNlib reference implementation [ANNlib].

---

### 5.3 `pareto_recent`

#### 5.3.1 Definition

Let `X_i ~ Pa(α)` (shape-α Pareto on `[1, ∞)`), then

$$t_i = (T - 1) - \left\lfloor (T - 1) \cdot \frac{\min(X_i, Q_{0.999}(X)) - 1}{Q_{0.999}(X) - 1} \right\rfloor.$$

**PDF of `X`**: `f(x) = α / x^{α+1}` for `x ≥ 1`.
**Mean of `X`**: `α / (α - 1)` for `α > 1` (else undefined).
**Heavy-tail.** Smaller α → heavier tail of *old* timestamps.

#### 5.3.2 Algorithm

```
1: X ← Pareto(α, size=n) + 1                # numpy returns Lomax form on [0,∞)
2: cap ← Quantile_{0.999}(X)
3: X ← min(X, cap); norm ← (X - 1) / (cap - 1)
4: t ← (T - 1) - floor((T - 1) * norm)
5: return clip(t, 0, T - 1)
```

The 0.999-quantile cap is a defensive truncation: raw Pareto draws can
produce a single extreme outlier that compresses all other points into one
bin. Truncating at the upper percentile keeps the empirical CDF well-spread.

#### 5.3.3 Complexity

Time `O(n)` plus the quantile (`O(n log n)` worst case, `O(n)` with
introselect; numpy uses `np.quantile` which is `O(n)`). Memory `O(n)`.

#### 5.3.4 Default parameter

`--pareto-alpha=1.2`. Justification: empirical fits of web page age and web
access frequency report α ∈ [1.0, 1.5] (Adamic & Huberman 2002
[AdamicHuberman2002]). Mitzenmacher's survey lists this as a typical Pareto
fit regime in computing systems [Mitzenmacher2004].

#### 5.3.5 Empirical context

- Page age in web archives [AdamicHuberman2002].
- File size distributions in network filesystems [DownyEtAl2002].
- Wikipedia edit frequency over article age [Wilkinson2008].

#### 5.3.6 Citations

[Pareto1896], [Mitzenmacher2004], [AdamicHuberman2002].

---

### 5.4 `lognormal_recent`

#### 5.4.1 Definition

Let `Y_i ~ N(μ, σ²)` and `X_i = exp(Y_i)`. Then

$$t_i = (T - 1) - \left\lfloor (T - 1) \cdot \frac{\min(X_i, Q_{0.999}(X))}{Q_{0.999}(X)} \right\rfloor.$$

**PDF of `X`**: `f(x) = (1 / (xσ √(2π))) · exp(-(ln x - μ)² / (2σ²))`.
**Mean of `X`**: `exp(μ + σ²/2)`. **Variance** `(exp(σ²) − 1) · exp(2μ + σ²)`.
Heavy-tailed for large σ; lighter than Pareto for the same intuitive shape.

#### 5.4.2 Algorithm

```
1: X ← LogNormal(μ, σ², size=n)             # numpy.Generator.lognormal
2: cap ← Quantile_{0.999}(X)
3: X ← min(X, cap); norm ← X / cap
4: t ← (T - 1) - floor((T - 1) * norm)
5: return clip(t, 0, T - 1)
```

#### 5.4.3 Complexity

Same as `pareto_recent`: `O(n)`.

#### 5.4.4 Default parameters

`--lognormal-mu=0.0`, `--lognormal-sigma=1.0`. Justification: log-normal fits
of video popularity decay [ChaPingali2009] estimate σ between 1 and 2; we
choose the lower end as a moderate-tail default. Limpert et al. survey the
ubiquity of log-normal in natural and social phenomena [LimpertEtAl2001].

#### 5.4.5 Empirical context

- Video popularity in user-generated content systems [ChaPingali2009].
- File-size distributions on the web [DownyEtAl2002].
- Inter-arrival times in many social activity streams [LimpertEtAl2001].

#### 5.4.6 Citations

[LimpertEtAl2001], [ChaPingali2009], [Mitzenmacher2004].

---

### 5.5 `topic_drift`

#### 5.5.1 Definition

Cluster vectors into `K` topics via mini-batch k-means (`c_i ∈ {1, …, K}`).
For each cluster `c`, draw

$$\mu_c \sim U(0, T), \qquad \sigma_c = |N(0, (\sigma_{\text{frac}} \cdot T)^2)| + 1.$$

Then for each point `i`:

$$t_i \sim N(\mu_{c_i}, \sigma_{c_i}^2), \quad
t_i \leftarrow \text{clip}(\lfloor t_i \rfloor, 0, T - 1).$$

This is the Gaussian variant of Wang & McCallum's "Topics over Time"
[WangMcCallum2006] generative model; their original uses a Beta time kernel.
Gaussian is chosen because the timeline is bounded and centered, and Beta's
support `[0,1]` parameterization would be conceptually identical after
rescaling — but Gaussian's `(μ, σ)` is more intuitive for a parameter
default.

#### 5.5.2 Algorithm

```
1: c ← MiniBatchKMeans(K, n_init=3, batch=4096).fit(sample of 80k from base).predict(base)
2: for c = 1 to K do
3:    μ_c ← Uniform(0, T)
4:    σ_c ← |N(0, (σ_frac · T)²)| + 1
5: for i = 1 to n do
6:    t_i ← clip(round(N(μ_{c_i}, σ_{c_i}²)), 0, T - 1)
7: return t
```

#### 5.5.3 Complexity

K-means fit on the 80k subsample is dominant: `O(80000 · d · K · iters)`,
typically ~1e9 FLOPs for d=128, K=128, 10 iters. Prediction on full `n` is
`O(n · d · K)`. Gaussian draws are `O(n)`. Total for `n=1e6`, `d=96`,
`K=128`: a few CPU-seconds; for `d=1536` (OpenAI), the prediction step
dominates at ~30s.

#### 5.5.4 Default parameters

| Param | Default | Justification |
|---|---|---|
| `--num-clusters` | `128` | order-of-magnitude consistent with topic counts in TM literature (50–500 in Blei & Lafferty 2006 [BleiLafferty2006]; 100 used in Wang & McCallum 2006). |
| `--sigma-frac` | `0.05` | ~5% of timeline width per topic; topic time-extents in Wang & McCallum 2006 concentrate near this fraction on their NIPS corpus. |

#### 5.5.5 Empirical context

- News topics with bounded life spans [WangMcCallum2006].
- Dynamic topic evolution in scientific publications [BleiLafferty2006].
- Fashion / visual embedding drift over years [HeMcAuley2016].

#### 5.5.6 Why this is the *primary* distribution

By Prop. 4.1, this is the *only* distribution in the catalogue under which
window membership ceases to be a uniform random subsample. Any benchmark
result that purports to distinguish RF-ANN baselines from each other should
include this distribution (or one like it). The independent univariate
distributions (5.2–5.4) cannot do so by themselves.

#### 5.5.7 Citations

[WangMcCallum2006], [BleiLafferty2006], [HeMcAuley2016].

---

### 5.6 `topic_burst`

#### 5.6.1 Definition

Cluster as in §5.5. For each cluster `c` simulate a univariate Hawkes process
[Hawkes1971] on `[0, T)` with exponential excitation kernel:

$$\lambda_c(t) = \mu + \alpha \sum_{t_j \in H_c(t)} e^{-\beta (t - t_j)}, \quad
H_c(t) := \{t_j : t_j < t,\;j\in\text{cluster }c\}.$$

The process is **sub-critical** iff the branching ratio `α / β < 1`
[BremaudMassoulie1996]. Each accepted event becomes a timestamp; if the
simulation under-produces relative to `|c|`, the remainder is padded
uniformly in `[0, T)`.

#### 5.6.2 Sampling: Ogata's thinning with incremental updates

```
1: events ← []; t_prev ← 0; S ← 0; lam_bar ← μ
2: while t_prev < T and |events| < cap do
3:    U ← Uniform(0,1]
4:    w ← -log(U) / lam_bar
5:    t ← t_prev + w
6:    if t ≥ T break
7:    S ← S · exp(-β · (t - t_prev))                # incremental decay
8:    λ_t ← μ + α · S                               # exact intensity at t
9:    if Uniform(0,1) · lam_bar ≤ λ_t then
10:       append t to events
11:       S ← S + 1                                  # absorb the new event
12:       lam_bar ← μ + α · S                        # bound after acceptance
13:    else
14:       lam_bar ← max(λ_t, μ)
15:    t_prev ← t
16: return events
```

**Correctness (Ogata 1981).** A candidate event proposed at rate `lam_bar`
(an upper bound on `λ_c(t)` over `[t_prev, ∞)`) and accepted with probability
`λ_c(t) / lam_bar` is a sample from the Hawkes process [Ogata1981]. The
incremental update at lines 7 and 11 maintains
`S(t) = Σ_{t_j ≤ t_prev} exp(-β(t_prev - t_j))` exactly:

- *Decay between events*: `S(t) = S(t_prev) · exp(-β · (t - t_prev))`
  because every term in the sum acquires an additional `exp(-β · w)` factor.
- *After acceptance*: `S(t) ← S(t) + 1` adds the new event at exactly the
  current time.

After acceptance, `λ_c(t)` jumps by `α` (the just-added event contributes
`α · 1`). All future intensities `λ_c(t')`, `t' > t` are bounded above by
`λ_c(t)` (sub-criticality + monotone decay between events), so the new bound
`μ + α · S` is valid.

#### 5.6.3 Complexity

`O(1)` work per candidate. Expected acceptance ratio ≈ stable, so total
candidates ≈ `O(|events|)`. Per-cluster cost `O(|cluster|)`. Total
`O(n)`, vs `O(n²/K)` for the naive sum-over-history form (which we avoid).

#### 5.6.4 Default parameters

| Param | Default | Justification |
|---|---|---|
| `--num-clusters` | `128` | same as `topic_drift`. |
| `--hawkes-mu` | `0.5` | base intensity, scales the "ambient" rate; chosen to seed a small but nonzero baseline. |
| `--hawkes-alpha` | `0.6` | jump per event. |
| `--hawkes-beta` | `1.0` | exponential decay; α/β = 0.6 → sub-critical near criticality, visible bursts without explosion (Reinhart 2018 [Reinhart2018] surveys parameter regimes). |

#### 5.6.5 Empirical context

- Tweet retweet cascades [ZhaoEtAl2015].
- Earthquake aftershocks (ETAS model) [Ogata1988].
- Financial trade arrival times [BacryEtAl2015].

#### 5.6.6 Citations

[Hawkes1971], [Ogata1981], [Ogata1988], [ZhaoEtAl2015], [BremaudMassoulie1996],
[Reinhart2018], [BacryEtAl2015].

---

### 5.7 `topic_seasonal`

#### 5.7.1 Definition

Cluster as in §5.5. For each cluster `c` with random phase
`φ_c ~ U(0, 2π)`, simulate a non-homogeneous Poisson process with intensity

$$\lambda_c(t) = \lambda_0 \cdot \left(1 + a \cdot \sin\!\left(\frac{2\pi t}{P} + \varphi_c\right)\right),
\quad \lambda_0 = \frac{|c|}{T}.$$

Different clusters peak at different times, modelling topical
diurnal/weekly seasonality. `a < 1` keeps `λ_c(t) > 0` everywhere.

#### 5.7.2 Sampling: Lewis-Shedler thinning

```
1: λ_peak ← λ_0 (1 + a)                     # an upper bound on λ_c(t)
2: candidates ← cumsum(Exponential(1/λ_peak, size ≈ λ_peak · T · 1.4))
3: candidates ← {t in candidates : t < T}
4: for each t in candidates:
5:    accept with probability λ_c(t) / λ_peak
6: pad / subsample to exactly |c| events; shuffle and assign to cluster points
```

**Correctness (Lewis-Shedler 1979).** Draw a homogeneous Poisson process at
rate `λ_peak` and thin each event with probability `λ_c(t) / λ_peak`. The
retained events form an NHPP with intensity `λ_c(t)` [LewisShedler1979].
Vectorising the cumulative-sum and acceptance over the candidate set yields
a one-shot numpy computation per cluster.

#### 5.7.3 Complexity

Per cluster: candidate generation `O(λ_peak · T)`. With `λ_0 ≈ |c|/T` and
`a < 1`, candidates ≈ `1.4 · (1+a) · |c|`. Total `O(n)`.

#### 5.7.4 Default parameters

| Param | Default | Justification |
|---|---|---|
| `--num-clusters` | `128` | same. |
| `--seasonal-period-frac` | `1/7` | one "week" per 7 sub-periods over `T`; mirrors weekly cycles in web access [KaragiannisEtAl2004]. |
| `--seasonal-amp` | `0.6` | rate varies in `[0.4 λ_0, 1.6 λ_0]`; consistent with the day-night amplitude reported in many web traces. |

#### 5.7.5 Empirical context

- Diurnal swings in Internet packet traffic [KaragiannisEtAl2004].
- Weekly cycles in Wikipedia traffic [Reinoso2009].
- Hour-of-day effects in social-media posting [GoldenbergEtAl2008].

#### 5.7.6 Citations

[LewisShedler1979], [Kingman1992], [KaragiannisEtAl2004].

---

## 6. Query / window workload

### 6.1 Query selection

`m` query IDs are sampled **uniformly without replacement** from
`{0, …, n-1}`. Query vectors are `base[qid]` (matches ANNlib & SeRF).

ANNlib's reference samples *with* replacement (`uniform_int_distribution` in
`generate_query_ids`); we differ to avoid duplicate-query bias in mean
recall.

### 6.2 Window-width semantics: rank vs raw-time

ANNlib's reference uses a **fixed raw-time width** (`-range W` in the C++
binary): every query has interval length exactly `W` on the timestamp axis.
Under non-uniform timestamp distributions, this makes per-query selectivity
**vary dramatically** — a fixed-width window in a dense region contains many
more points than in a sparse region.

This generator uses **target rank width**: for window-width parameter `W`,
each query interval is constructed to contain ≈ `W` points regardless of the
underlying distribution. Concretely, given sorted timestamps
`s_0 ≤ s_1 ≤ … ≤ s_{n-1}`, a query with center rank `r` becomes
`[s_{max(0, r - W/2)}, s_{min(n-1, r + W/2)}]`.

**Why this matters.** Holding selectivity fixed across distributions is what
makes recall and QPS comparable across them. Fair benchmark comparison
requires selectivity-controlled queries.

For direct ANNlib parity, a `--width-mode=time` switch could be added that
generates raw-time windows; we have not done so to avoid hiding the
selectivity drift that is the real source of difficulty.

### 6.3 Center-bias models

The CLI accepts `--center-bias` ∈ `{uniform, recent}`:

- `uniform`: center rank `~ U(0, n)`.
- `recent`: center rank `~ ⌊n · Beta(5, 1)⌋`. Beta(5, 1) is heavily
  right-skewed (mode at 1.0, mean 5/6) — most queries cluster in the
  *recent* half of the timeline. This models recency-biased user query
  workloads documented in recommender systems
  [KonstanRiedl2012, KoyaEtAl2009].

We deliberately keep window **width** distribution-free (single rank-width
`W` per file) to keep cross-distribution comparison clean. To study width
heterogeneity, generate multiple `range<W>.*` files at different `W`s — the
wrappers consume each separately.

---

## 7. Ground truth: brute force with windowing

### 7.1 Statement

For each query `q ∈ {1, …, m}` with vector `v_q := x_{qid_q}` and window
`[lo_q, hi_q]`, the ground truth is the top-`k` set under the chosen
distance, restricted to points whose timestamps fall in the window:

$$\text{GT}_q = \arg\!\min_{i :\; lo_q \le t_i \le hi_q}^{\;k}\;
\rho(v_q, x_i)$$

where `ρ(u, v) = ||u - v||²₂` for `--dist L2` and
`ρ(u, v) = -⟨u/||u||, v/||v||⟩` for `--dist angular`. Both forms are
rank-equivalent to true Euclidean / cosine distance and yield identical
top-`k` sets.

### 7.2 Implementation: chunked materialised distance matrix

For each `chunk` of 256 queries:

1. Compute the `(chunk, n)` distance matrix `D`:
   - `L2`: `D[j, i] = ||v_q[j]||² + ||x_i||² - 2 ⟨v_q[j], x_i⟩` (numerically
     stable when both terms are float32 and the dataset isn't pathological;
     for problematic cases the cross-term can be computed via `cdist` with
     a small constant-factor penalty).
   - `angular`: `D[j, i] = -⟨v_q[j], x_i⟩` (both sides L2-normalised first).
2. For each query `j` and each width `W`:
   - Build the window mask `mask = (lo ≤ t) ∧ (t ≤ hi)`.
   - Extract `D[j, mask]`, `argpartition` top-`k`, sort.

### 7.3 Complexity and memory

Time: dominant cost is the matmul `qc @ b.T`, total `O(m · n · d)` across
all chunks ≈ `1e3 · 1e6 · d` FLOPs. On the 1M datasets this is:

| dataset | `d` | matmul FLOPs | wall-clock (single-node, AVX2) |
|---|---|---|---|
| bigann (uint8 → float32) | 128 | 1.3e11 | ~10s |
| Yandex-DEEP | 96 | 9.6e10 | ~8s (observed) |
| Cohere | 1024 | 1.0e12 | ~80s |
| OpenAI | 1536 | 1.5e12 | ~120s |

Memory: peak `chunk · n · 4` bytes ≈ 1 GB (chunk=256, n=1e6). Plus the base
vectors `n · d · 4` bytes (≈ 6 GB for OpenAI).

### 7.4 Padding convention

If `|window_q| < k`, the GT row is padded with `0xFFFFFFFF`. This matches
ANNlib's `numeric_limits<uint32_t>::max()` writer convention
(`timerange_dataset.hpp:164`) and is decoded as `-1` (int32) by the Python
wrapper at `lib/common.py:274`.

---

## 8. Recommended experimental matrix and statistical reporting

### 8.1 Minimum viable benchmark

Three distributions × four selectivities × one center bias × one seed:

```
P(t):         uniform   topic_drift   topic_burst
W:            100   1000   10000   100000           (target rank widths)
center_bias:  uniform
seed:         42
```

- `uniform`: lower-bound control (everything looks the same).
- `topic_drift`: realistic correlated workload — the only one of these three
  whose results actually depend on baseline algorithm choice.
- `topic_burst`: workload-position-sensitivity stressor.

Total: 12 configurations per dataset; with 4 datasets, 48 runs.

### 8.2 Full ablation matrix

Add to the above:

- `pareto_recent`, `lognormal_recent` (controls — should match `uniform`).
- `topic_seasonal` (periodic stressor).
- `--center-bias recent` rows (workload skew).
- Multiple seeds (recommended ≥ 3 for the *primary* distribution).
- Sensitivity sweep of `--num-clusters ∈ {50, 128, 500}` for `topic_*` (see
  Wang & McCallum 2006 §6 on topic-count sensitivity).

### 8.3 Statistical reporting

When reporting recall/QPS for a baseline `B` on a dataset `D` with
distribution `P` at selectivity `W`:

- Run with `≥ 3` independent `--seed` values (changes both timestamp draws
  and query sampling).
- Report **mean ± std** across seeds. Single-seed numbers are not adequate
  for distribution-comparison claims.
- For QPS, run a warm-up pass first.
- For recall, the GT itself is exact (brute force), so recall variance comes
  only from the index's randomness; if the baseline is deterministic given
  build params, multiple seeds primarily inflate workload variation.

### 8.4 Putting numbers in a table

A defensible per-dataset table reports, for each (distribution, baseline, W):

| baseline | W=100 recall@10 | W=100 QPS | … | W=100000 recall@10 | W=100000 QPS |
|---|---|---|---|---|---|

with the agreement of all three control distributions reported once as a
sanity row, and the spread across seeds shown via error bars or
sub-superscripts.

---

## 9. Reproducibility checklist

- [ ] `--seed` recorded in `manifest.txt` per artifact directory.
- [ ] Python / numpy / scikit-learn version pinned (record `pip freeze`).
- [ ] BLAS thread count fixed (`export OPENBLAS_NUM_THREADS=...`); BLAS
      non-determinism in the matmul does not affect GT (top-k is robust
      to ULP-level perturbations) but may affect timing reproducibility.
- [ ] Base dataset file's SHA-256 recorded (one-time, separately).
- [ ] ANNlib submodule commit SHA recorded (relevant if you also publish
      ANNlib-generated baselines for comparison).

A minimal reproduction recipe goes:

```bash
git checkout <commit>
git -C ANNlib checkout <annlib-sha>
source ~/.venv312/bin/activate
pip install numpy==<X> scikit-learn==<Y>
export OPENBLAS_NUM_THREADS=64
bash scripts/temporal_ann_baselines/run_synth_timestamps.sh
```

---

## 10. Validation and sanity checks

After running, validate per-artifact-directory with:

```python
import sys
sys.path.insert(0, 'scripts/temporal_ann_baselines')
from lib.common import load_timerange_artifacts

a = load_timerange_artifacts('/path/to/artifacts_dir',
                             ranges=[100, 1000, 10000, 100000])

# 1. Shapes
assert a.point_timestamps.shape == (1_000_000,)
assert a.query_ids.shape == (1_000,)
for w in [100, 1000, 10000, 100000]:
    assert a.query_ranges[w].shape == (1_000, 2)
    assert a.groundtruth[w].shape == (1_000, 10)

# 2. Query IDs are valid base indices
assert (a.query_ids < 1_000_000).all() and (a.query_ids >= 0).all()

# 3. Each range is non-empty (lo ≤ hi)
for w, r in a.query_ranges.items():
    assert (r[:, 0] <= r[:, 1]).all()

# 4. Selectivity is in expected range
ts = a.point_timestamps
for w, r in a.query_ranges.items():
    hits = ((ts[:, None] >= r[:, 0]) & (ts[:, None] <= r[:, 1])).sum(axis=0)
    print(f'range{w}: mean hits = {hits.mean():.1f}, expected ≈ {w}')

# 5. GT IDs in window
INVALID = 2**32 - 1
for w in a.groundtruth:
    gt = a.groundtruth[w]; r = a.query_ranges[w]
    for q in range(len(gt)):
        ids = gt[q][gt[q] != INVALID]
        if ids.size == 0:
            continue
        assert ((ts[ids] >= r[q, 0]) & (ts[ids] <= r[q, 1])).all(), \
               f'GT id out of window at q={q}'
```

The end-to-end smoke test (`smoke_test_synth.py`) automates these checks
plus brute-force-verified GT correctness on a synthetic 2k × 32 dataset; we
recommend running it any time the script is modified.

### 10.1 Cross-control invariance check

Per Prop. 4.1, three histograms — recall@k vs W for `uniform`,
`pareto_recent`, and `lognormal_recent` on the same baseline — should be
statistically indistinguishable. Visible disagreement across these three
indicates an implementation bug.

---

## 11. Limitations and threats to validity

1. **Rank-compression equivalence (re-stated).** Three of the six
   distributions are statistically equivalent post-compression; they exist
   as controls. A paper that *only* reported numbers from these would be
   misleading.

2. **K-means uses Euclidean geometry.** For angular datasets (OpenAI,
   Cohere), `topic_*` distributions cluster in the original (unnormalised)
   space, not on the unit sphere. This is principled when the embedding
   model already produces near-norm-1 vectors (both OpenAI and Cohere
   embeddings are L2-normalised by default) but does not generalise to
   arbitrary embeddings. Pre-normalisation before clustering can be added
   with one line; we did not enable it by default because it would silently
   diverge from how the baselines themselves consume the same data (which
   is normalised only at query time).

3. **Self-queries.** Query vectors are sampled rows of the base set,
   inherited from ANNlib and SeRF convention. For held-out evaluation,
   supply a separate query file and adapt the generator.

4. **Window widths are in rank, not time.** See §6.2. This is a deliberate
   methodological choice for cross-distribution comparability and a
   *deviation from ANNlib parity*. It must be disclosed.

5. **No baseline-attribute correlation beyond k-means.** Real corpora often
   exhibit fine-grained vector-time correlation (e.g. concept drift within
   a topic). K-means with `K=128` gives a coarse approximation. Higher `K`
   or hierarchical clustering would refine this; we keep the default coarse
   for tractable defaults.

6. **Single-machine BLAS GT.** For datasets beyond ~10M points, the
   `O(m · n · d)` matmul becomes inconvenient; switch to faiss-gpu or
   distributed brute force. The artifact format does not need to change.

7. **No claim of being the "right" distribution.** The catalogue is a
   *menu of canonical distributions backed by published generative models*,
   not an empirical fit to any specific corpus. For domain-specific
   benchmarks, fit the parameters to your own data (e.g. fit `α` of
   `pareto_recent` to your real-corpus age histogram, or `(μ, σ)` of
   `lognormal_recent` to your video popularity decay).

---

## 12. Bibliography (BibTeX)

```bibtex
@misc{ANNlib,
  title  = {{ANNlib} — UCRParlay parallel ANN library},
  author = {UCRParlay},
  note   = {\url{https://github.com/ucrparlay/ANNlib}, branch
            \texttt{timestamp-feature}, file
            \texttt{test/generate\_dataset/point\_timerange\_ann/timerange\_dataset.hpp}}
}

@book{Pareto1896,
  author    = {Pareto, Vilfredo},
  title     = {Cours d'\'Economie Politique},
  publisher = {F. Rouge},
  year      = {1896}
}

@article{AdamicHuberman2002,
  author  = {Adamic, Lada A. and Huberman, Bernardo A.},
  title   = {{Z}ipf's law and the {I}nternet},
  journal = {Glottometrics},
  volume  = {3},
  pages   = {143--150},
  year    = {2002}
}

@article{Mitzenmacher2004,
  author  = {Mitzenmacher, Michael},
  title   = {A brief history of generative models for power law and lognormal distributions},
  journal = {Internet Mathematics},
  volume  = {1},
  number  = {2},
  pages   = {226--251},
  year    = {2004}
}

@article{LimpertEtAl2001,
  author  = {Limpert, Eckhard and Stahel, Werner A. and Abbt, Markus},
  title   = {Log-normal distributions across the sciences: keys and clues},
  journal = {BioScience},
  volume  = {51},
  number  = {5},
  pages   = {341--352},
  year    = {2001}
}

@article{ChaPingali2009,
  author  = {Cha, Meeyoung and Kwak, Haewoon and Rodriguez, Pablo and Ahn,
             Yong-Yeol and Moon, Sue},
  title   = {Analyzing the video popularity characteristics of large-scale
             user generated content systems},
  journal = {IEEE/ACM Transactions on Networking},
  volume  = {17},
  number  = {5},
  pages   = {1357--1370},
  year    = {2009}
}

@article{DownyEtAl2002,
  author  = {Downey, Allen B.},
  title   = {The structural cause of file size distributions},
  journal = {ACM SIGMETRICS Performance Evaluation Review},
  volume  = {29},
  number  = {1},
  pages   = {361--370},
  year    = {2002}
}

@inproceedings{Wilkinson2008,
  author    = {Wilkinson, Dennis M.},
  title     = {Strong regularities in online peer production},
  booktitle = {ACM Conference on Electronic Commerce (EC)},
  pages     = {302--309},
  year      = {2008}
}

@inproceedings{WangMcCallum2006,
  author    = {Wang, Xuerui and McCallum, Andrew},
  title     = {Topics over time: a non-{M}arkov continuous-time model of topical trends},
  booktitle = {Proc. ACM SIGKDD},
  pages     = {424--433},
  year      = {2006}
}

@inproceedings{BleiLafferty2006,
  author    = {Blei, David M. and Lafferty, John D.},
  title     = {Dynamic topic models},
  booktitle = {Proc. ICML},
  pages     = {113--120},
  year      = {2006}
}

@inproceedings{HeMcAuley2016,
  author    = {He, Ruining and McAuley, Julian},
  title     = {Ups and downs: modeling the visual evolution of fashion trends
               with one-class collaborative filtering},
  booktitle = {Proc. WWW},
  pages     = {507--517},
  year      = {2016}
}

@article{Hawkes1971,
  author  = {Hawkes, Alan G.},
  title   = {Spectra of some self-exciting and mutually exciting point processes},
  journal = {Biometrika},
  volume  = {58},
  number  = {1},
  pages   = {83--90},
  year    = {1971}
}

@article{Ogata1981,
  author  = {Ogata, Yosihiko},
  title   = {On {Lewis}' simulation method for point processes},
  journal = {IEEE Transactions on Information Theory},
  volume  = {27},
  number  = {1},
  pages   = {23--31},
  year    = {1981}
}

@article{Ogata1988,
  author  = {Ogata, Yosihiko},
  title   = {Statistical models for earthquake occurrences and residual
             analysis for point processes},
  journal = {Journal of the American Statistical Association},
  volume  = {83},
  number  = {401},
  pages   = {9--27},
  year    = {1988}
}

@inproceedings{ZhaoEtAl2015,
  author    = {Zhao, Qingyuan and Erdogdu, Murat A. and He, Hera Y. and
               Rajaraman, Anand and Leskovec, Jure},
  title     = {{SEISMIC}: a self-exciting point process model for predicting
               tweet popularity},
  booktitle = {Proc. ACM SIGKDD},
  pages     = {1513--1522},
  year      = {2015}
}

@article{BremaudMassoulie1996,
  author  = {Br\'emaud, Pierre and Massouli\'e, Laurent},
  title   = {Stability of nonlinear {H}awkes processes},
  journal = {Annals of Probability},
  volume  = {24},
  number  = {3},
  pages   = {1563--1588},
  year    = {1996}
}

@misc{Reinhart2018,
  author = {Reinhart, Alex},
  title  = {A review of self-exciting spatio-temporal point processes and their applications},
  year   = {2018},
  note   = {Statistical Science 33(3): 299--318}
}

@article{BacryEtAl2015,
  author  = {Bacry, Emmanuel and Mastromatteo, Iacopo and Muzy, Jean-Fran\c{c}ois},
  title   = {{H}awkes processes in finance},
  journal = {Market Microstructure and Liquidity},
  volume  = {1},
  number  = {1},
  pages   = {1550005},
  year    = {2015}
}

@article{LewisShedler1979,
  author  = {Lewis, P. A. W. and Shedler, G. S.},
  title   = {Simulation of nonhomogeneous {P}oisson processes by thinning},
  journal = {Naval Research Logistics Quarterly},
  volume  = {26},
  number  = {3},
  pages   = {403--413},
  year    = {1979}
}

@book{Kingman1992,
  author    = {Kingman, J. F. C.},
  title     = {{P}oisson Processes},
  publisher = {Oxford University Press},
  series    = {Oxford Studies in Probability},
  volume    = {3},
  year      = {1992}
}

@inproceedings{KaragiannisEtAl2004,
  author    = {Karagiannis, Thomas and Molle, Mart and Faloutsos, Michalis
               and Broido, Andre},
  title     = {A nonstationary {P}oisson view of {I}nternet traffic},
  booktitle = {Proc. IEEE INFOCOM},
  pages     = {1558--1569},
  year      = {2004}
}

@article{Reinoso2009,
  author  = {Reinoso, Antonio J. and Mu\~noz-Mansilla, Rocio and Herraiz,
             Israel and Ortega, Felipe},
  title   = {Characterization of the {W}ikipedia traffic},
  journal = {Internet Research},
  year    = {2009}
}

@inproceedings{GoldenbergEtAl2008,
  author    = {Golder, Scott A. and Wilkinson, Dennis M. and Huberman, Bernardo A.},
  title     = {Rhythms of social interaction: messaging within a massive online network},
  booktitle = {Communities and Technologies (C\&T)},
  pages     = {41--66},
  year      = {2008}
}

@article{KonstanRiedl2012,
  author  = {Konstan, Joseph A. and Riedl, John},
  title   = {Recommender systems: from algorithms to user experience},
  journal = {User Modeling and User-Adapted Interaction (UMUAI)},
  volume  = {22},
  number  = {1--2},
  pages   = {101--123},
  year    = {2012}
}

@article{KoyaEtAl2009,
  author  = {Koren, Yehuda},
  title   = {Collaborative filtering with temporal dynamics},
  journal = {Communications of the ACM},
  volume  = {53},
  number  = {4},
  pages   = {89--97},
  year    = {2010}
}

@article{Cox1955,
  author  = {Cox, D. R.},
  title   = {Some statistical methods connected with series of events},
  journal = {Journal of the Royal Statistical Society. Series B},
  volume  = {17},
  number  = {2},
  pages   = {129--164},
  year    = {1955}
}

@article{MandelbrotVanNess1968,
  author  = {Mandelbrot, Benoit B. and Van Ness, John W.},
  title   = {Fractional {B}rownian motions, fractional noises and applications},
  journal = {SIAM Review},
  volume  = {10},
  number  = {4},
  pages   = {422--437},
  year    = {1968}
}

@inproceedings{ZuoEtAl2024,
  author    = {Zuo, Chaoji and Qiao, Miao and Zhou, Wenchao and Li, Feifei and Deng, Dong},
  title     = {{SeRF}: segment graph for range-filtering approximate nearest neighbor search},
  booktitle = {Proc. ACM Management of Data (SIGMOD)},
  volume    = {2},
  number    = {1},
  year      = {2024}
}

@article{XuEtAl2024,
  author  = {Xu, Yuexuan and others},
  title   = {{iRangeGraph}: improvising range-dedicated graphs for range-filtering nearest neighbor search},
  journal = {Proc. ACM Management of Data},
  year    = {2024},
  note    = {Also \url{https://arxiv.org/abs/2409.02571}}
}

@misc{DSG2025,
  author = {Rutgers Database Lab},
  title  = {Dynamic Range-Filtering Approximate Nearest Neighbor Search (Dynamic Segment Graph)},
  year   = {2025},
  note   = {VLDB 2025}
}

@misc{UNIFY2024,
  author = {SJTU DB Group},
  title  = {{UNIFY}: a unified index for range-filtered approximate nearest neighbor search},
  year   = {2024}
}

@misc{EngelsEtAl2024,
  author = {Engels, Joshua and others},
  title  = {Approximate nearest neighbor search with window filters},
  year   = {2024},
  note   = {\url{https://arxiv.org/abs/2402.00943}}
}
```

---

*End of document. For runtime details consult the docstrings in
`synth_timestamps.py`; for the end-to-end validation harness see
`smoke_test_synth.py`.*

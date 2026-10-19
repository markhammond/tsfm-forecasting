# Changelog

## 0.2.0 — unreleased

### Tsfm.Forecasting.T0 (Apache-2.0) — new

- Inference for The Forecasting Company's **t0-alpha** (102M) and **t0-beta** (256M), ported
  from `tfc-t0` 0.6.0: causal scaler, patch encoder, alternating time/group transformer
  with xPos rotary embedding, residual decoder, monotone quantile head.
- `T0Forecaster` takes the TimesFM shape: `ForecastJoint` over variate-major series with
  known-future flags and a target index, and `Forecast` over OHLCVA for forward returns.
  Unlike TimesFM, any context length is accepted (left-padded onto the patch grid and
  masked), and up to 1024 steps are decoded in one pass.
- Each checkpoint's scaler convention is read from its `config.json`. t0-beta's differs
  from alpha's, and running it under alpha's loads cleanly and forecasts worse.
- Verified stage by stage against the reference, both checkpoints, including the
  quickstart notebook's electricity-price and M4 examples (`tests/Tsfm.Forecasting.T0.Parity`,
  `reference/dump_t0_parity.py`).
- Weights are Apache-2.0 but not packaged: each exceeds nuget.org's 250 MB limit.
  `scripts/fetch-t0-checkpoint.sh` fetches pinned revisions and checks their SHA-256.
- Not ported: autoregressive rollout past 1024 steps, quantile interpolation and tail
  extrapolation to levels off the trained grid, and the optional symmetric averaging,
  clipping and integer snapping.

- Weights are not bundled. The README gives pinned download commands with checksums for
  projects that only reference the package, and `Load` names them when files are missing.

### Batching, across all three models

- `IQuantileForecaster` (new, in `Tsfm.Forecasting`): the shared `ForecastJoint` /
  `Forecast` / `Quantiles` shape, plus `ForecastJointBatch` and `ForecastBatch` over many
  entries sharing a horizon, written `[entry][step][quantile]` into a caller-sized span
  (`BatchOutputLength`). Entries are `JointSeries` / `ReadOnlyMemory<float>`, so anchors
  sliced from one history copy nothing. `TimesFmForecaster` and `T0Forecaster` implement it.
- t0 batches any entries in one pass — differing variates, contexts and known-future
  patterns — each its own attention group, shorter contexts left-padded and masked. Entries
  are ordered by context before being cut into passes of `MaxRowsPerPass` series. Verified
  against the reference in the parity harness.
- TimesFM stacks entries of one shape on the model's batch axis, `MaxBatchPerPass` at a
  time. The single calls now run as a batch of one; outputs are bit-identical to 0.1.3.
- `KronosForecaster.InferBatch` forecasts the final window of many series in one pass of
  `batch × rollouts` rows; row `i` equals the last row `InferPath` gives for series `i`
  alone, sampling included. `Infer` and `InferPath` share the refactored loop and are
  bit-identical to 0.1.3.
- `KronosForecaster.Seed` makes the sampling seed a host policy over the anchor bar and a
  series key (`KronosSeries.SeriesKey`, or `seriesKey:` on `Infer`/`InferPath`).
  `SamplingSeeds.AnchorTime`, the default, is the 0.1.x behaviour: series anchored on one bar
  share draws (common random numbers). `PerSeries` gives each key its own stream, and
  `Salted` derives a fresh reproducible run from any policy. Batch invariance holds under
  all of them; the default is bit-identical to 0.1.3.

### Packaging

All six packages move to 0.2.0: the core gains the interface the model packages implement,
and a project reference packs an exact-version dependency.

### Samples and testing

- `samples/GroceryCovariates` runs on TimesFM or t0 (`-- t0-beta`) through the interface.
- Batch tests for TimesFM and t0 run on small random-init models, so they bite in CI
  without weights; Kronos's run with the integration tests.
- `Tsfm.Forecasting.T0.Tests` checks structural invariants on a tiny random-init model, so
  they run in CI without weights: a target never reads its own future, known-future rows
  never read targets, time attention is causal, any context length is accepted. Behaviour
  on the published checkpoints is covered too, gated on their presence.

## 0.1.3 — 2026-09-02

### Joint forecasting with known-future covariates

- `TimesFmForecaster.ForecastJoint` takes several series at once (up to 32), attends over
  them jointly, and lets any of them carry values known into the future. Series are
  normalised per variate, so mixing a price in the thousands with a flag in {0,1} needs no
  scaling. There is no variate identity or ordering — the model sees anonymous series and
  infers their relationship from co-movement in the supplied window.
- A variate marked known-future is by construction not a forecast target; marking the
  target is refused, since that would hand a series its own future.
- The known future is gathered by rolling whole patches forward, so the context must be a
  positive multiple of the patch length and leave at least one patch after it. Both are
  now validated rather than producing a silently wrong anchor.
- The variate ceiling is read from the checkpoint rather than hardcoded: it sits in the
  same config block as the layer and head counts, so another checkpoint may declare a
  different figure. `TimesFmForecaster.MaxVariates` reports what the loaded one states.
- `samples/GroceryCovariates` reproduces the scenario from Google's covariates notebook —
  ice cream and sunscreen, with temperature and promotion known for both weeks. Its
  numbers are NOT comparable: the notebook uses the XReg path, which regresses covariates
  on forecast residuals, whereas 3.0 attends over them as variates.

### Packaging

Only `Tsfm.Forecasting.TimesFm` changed. The other four are republished at the same
version because a project reference packs an exact-version dependency, so TimesFm 0.1.3
requires a core 0.1.3 to resolve against.

### Testing

- `Tsfm.Forecasting.TimesFm.IntegrationTests` covers the joint API, including that
  changing a known-future variate actually moves the forecast, and that identical input
  does not. These are **local only**: the TimesFM checkpoint is ~1.2 GB and
  non-commercially licensed, so CI never fetches it and the tests pass vacuously there.
  CI runs them regardless, which proves they compile and start.

## 0.1.2 — 2026-09-02

Names two things that were previously literals a caller had to know.

- `KronosForecaster.Channels` replaces the bare `6` at every call site, and carries the
  row-major layout, the channel order and the amount derivation in its own documentation.
- `ICheckpoint.MaxContext` states the longest context a checkpoint can attend over: 2048
  for Kronos-mini, 512 for Kronos-small and -base. It varies by checkpoint, so it is not
  a constant of the architecture.
- `Infer` now refuses a context beyond that limit. Upstream truncates silently, which
  reads as a working call that quietly ignored the bars beyond the ceiling.
- The sample takes its context from the checkpoint rather than a hard-coded number, and
  selects its device through `DeviceType` rather than a string.

No change to inference; the ports remain parity-verified.

## 0.1.1 — 2026-09-02

### Projected paths

- `KronosForecaster.InferPath` writes the projected OHLCVA as `rows x horizon x 6`,
  sized by `PathLength`. The decode always produced all six channels; only the close
  was surfaced.
- Rollouts are reduced per channel per step by a caller-supplied `RolloutAggregator`
  (`ReadOnlySpan<float> -> float`), which is the shape the reference uses internally
  (`np.mean(preds, axis=1)`). `Mean` is the default, matching it; `Median`,
  `AtQuantile` and `Rollout` are provided. Samples are ordered by rollout index.
- Paths are frequently not well-formed candles, and this is the model rather than the
  port: channels decode independently with nothing tying `high` to `close`. A single
  rollout broke ordering in roughly 30% to 70% of candles depending on the checkpoint,
  smaller ones being worse; averaging across rollouts reduces that severalfold, cancelling
  the noise rather than enforcing anything.
  The reference does no post-processing either, and its headline example runs
  `sample_count=1`. Clamp if you need valid candles.

### Documentation reaches consumers

- `GenerateDocumentationFile` was never set, so 0.1.0 shipped no `.xml` and gave
  consumers no IntelliSense beyond signatures. All five packages now carry it.
- `Infer` documents every parameter, including which channels are filled rather than
  ignored — the model reads six and has no mask, and the reference derives amount as
  `volume * mean(open, high, low, close)`.
- Both forecasters record that inference is not thread safe.

### Allocation and API shape

- Per-window scratch is rented once and passed in pre-sliced rather than allocated per
  window, and the per-batch `ToArray()` in the sampler is gone.
- **Breaking:** `WriteStamp` takes `Span<float>` rather than `float[]`. Source-compatible
  for array callers, binary-breaking against 0.1.0.

The Kronos and TimesFM inference paths are otherwise unchanged and still parity-verified.

## 0.1.0 — 2026-08-30

First release of the family.

Supersedes `Kronos.Forecasting` 0.1.0, which was published under the previous name and
is deprecated in favour of `Tsfm.Forecasting.Kronos`. The Kronos code is unchanged; only
the package identity and namespace moved.

### Tsfm.Forecasting (MIT)

- `Safetensors` reader — checkpoints load without Python at build or run time
- `ICheckpoint`: weights as streams, from embedded resources or a directory

### Tsfm.Forecasting.Kronos (MIT)

- Kronos tokenizer (encode and decode) and autoregressive model, ported to TorchSharp
- `Tsfm.Forecasting.Kronos.Weights.Small` and `.Mini`, checkpoints embedded at pinned
  revisions
- Verified stage by stage against the reference on CPU and Metal

### Tsfm.Forecasting.TimesFm (Apache-2.0)

- TimesFM 3.0 inference: patch embedding, the mixing transformer stack (sequence
  attention, variate attention, feed-forward), quantile head, and the surrounding
  preprocessing — cumulative running statistics, RevIN, future-covariate rolling
- OHLCVA supplied as variates; one forward pass yields 64 steps at nine quantiles
- Verified stage by stage against the reference on CPU and Metal
- **No weights, and none can be published.** The checkpoints are non-commercial; see
  README and NOTICE

Licences differ per package because the ports derive from differently licensed sources.
Neural modules are deliberately not shared between the two adapters: RmsNorm, RoPE and
attention differ in exactly the details that parity checking exposed, and unifying them
would put a verified port at risk to save little.

Known limitations, all deliberate and documented in the README: neither model has a
key-value cache, so attention is recomputed per decode step to preserve checkable
parity; results are not bit-identical across devices; and Kronos's sampling stream
diverges from its reference in order to be batch-invariant, which the reference is not.

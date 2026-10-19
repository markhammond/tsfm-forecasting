# Tsfm.Forecasting

Native .NET inference for time-series foundation models, on CPU or GPU via
TorchSharp. Currently [Kronos](https://github.com/shiyu-coder/Kronos), [TimesFM 3.0](https://github.com/google-research/timesfm)
and [t0](https://huggingface.co/theforecastingcompany/t0-beta) (t0-alpha, t0-beta).

[![ci](https://github.com/markhammond/tsfm-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/markhammond/tsfm-forecasting/actions/workflows/ci.yml)

Targets .NET 10

NuGet packages:

| Package | Licence | |
|---|---|---|
| Tsfm.Forecasting [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.svg)](https://www.nuget.org/packages/Tsfm.Forecasting) | MIT | safetensors reader, checkpoint abstraction |
| Tsfm.Forecasting.Kronos [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.Kronos.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.Kronos) | MIT | tokenizer, model, forecaster |
| Tsfm.Forecasting.Kronos.Weights.Small [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.Kronos.Weights.Small.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.Kronos.Weights.Small) | MIT | Kronos-small (24.7M) + Tokenizer-base |
| Tsfm.Forecasting.Kronos.Weights.Mini [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.Kronos.Weights.Mini.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.Kronos.Weights.Mini) | MIT | Kronos-mini (4.1M) + Tokenizer-2k |
| Tsfm.Forecasting.TimesFm [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.TimesFm.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.TimesFm) | **Apache-2.0** | TimesFM 3.0 — **no weights** |
| Tsfm.Forecasting.T0 [![NuGet](https://img.shields.io/nuget/v/Tsfm.Forecasting.T0.svg)](https://www.nuget.org/packages/Tsfm.Forecasting.T0) | **Apache-2.0** | t0-alpha, t0-beta — open weights, fetched separately |

Refer to respective package for license and conditions of use (if any).

## Why this exists

None of these models is a text LLM, so existing .NET inference runtimes cannot load them.
Kronos pairs a Binary Spherical Quantization autoencoder with a decoder that has a
hierarchical two-subtoken embedding, a dependency-aware cross-attention layer and two
conditional heads. GGUF has no representation for any of that, which rules out
llama.cpp-based runtimes; the alternatives that *do* reach Apple's GPU load GGUF only.
The architectures had to be ported.

## Kronos — batteries included

```
dotnet add package Tsfm.Forecasting.Kronos
dotnet add package Tsfm.Forecasting.Kronos.Weights.Small   # or .Mini
dotnet add package TorchSharp-cpu                          # or a CUDA backend
```

```csharp
using Tsfm.Forecasting.Kronos;
using Tsfm.Forecasting.Kronos.Weights;
using static TorchSharp.torch;

using var forecaster = KronosForecaster.Load(KronosSmall.Instance, new Device("mps"));

var rows = KronosForecaster.OutputCount(barCount, contextBars: 384);
Span<float> lean = new float[rows];
Span<int>   upCount = new int[rows];

forecaster.Infer(
    ohlcva,          // row-major [L x 6]: open, high, low, close, volume, amount
    barTimeMs,       // [L] Unix ms, one per bar
    lean, upCount, dispersion: default,
    contextBars: 384, horizon: 1, rollouts: 30,
    greedy: false, temperature: 1f, topP: 1f);
```

### Supplying fewer than six channels

The model always reads six channels and has no mask, so an absent one is **filled, not
ignored** — whatever you supply is read as data. There is a single entry point rather
than `ohlc`/`ohlcv` overloads, because choosing the filler is a modelling decision and
belongs at the call site. These are the fills the reference implementation uses:

| You have | volume | amount |
|---|---|---|
| OHLC | `0` | `0` |
| OHLCV | as given | `volume * mean(open, high, low, close)` |
| OHLCVA | as given | as given |

Note the mean of all four prices, not the close. The reference rejects NaN in any of the
six rather than treating it as absent, so fill explicitly.

Checkpoints are embedded as assembly resources, so nothing resolves through
configuration or the filesystem — which is what lets this sit behind a consumer
forbidden from reading its environment. Buffers are caller-supplied and sized from
`OutputCount`; a mismatch throws with the expected count rather than silently
misaligning every row.

## TimesFM — bring your own weights

> **The weights are not open source and are not distributed here.** TimesFM 3.0
> checkpoints are published under the **TimesFM Non-Commercial License v1.0**: research
> and evaluation only, with revenue-generating activity and production deployment
> expressly forbidden. This package's Apache-2.0 licence grants no rights in them
> whatsoever. Commercial use needs terms from Google.

```
dotnet add package Tsfm.Forecasting.TimesFm
dotnet add package TorchSharp-cpu
./scripts/fetch-timesfm-checkpoint.sh    # ~1.2 GB — read the licence first
```

```csharp
using Tsfm.Forecasting.TimesFm;

var forecaster = TimesFmForecaster.Load("checkpoints/timesfm-3.0-pytorch", new Device("mps"));
double[,] q = forecaster.Forecast(ohlcva, horizon: 4);   // [step, quantile]
```

One forward pass yields 64 steps at all nine quantiles, so a full predictive interval
costs no more than a point forecast. Every OHLCVA channel is supplied as a variate and
marked a target: preprocessing masks future covariate slots only for target variates, so
a channel left non-target would be handed its own future values.

## t0 — open weights, fetched

[t0](https://arxiv.org/abs/2609.24559) from The Forecasting Company is a decoder-style patch
transformer alternating causal time attention with variate (group) attention, with native
known-future covariates. Both checkpoints are **Apache-2.0**. They are fetched rather than
packaged only because each exceeds nuget.org's 250 MB limit.

| Checkpoint | Params | Quantiles | Scaler eps |
|---|---|---|---|
| t0-alpha | 102M | 5 (0.1 … 0.9) | 0.1, variance offset |
| t0-beta | 256M | 21 (0.01 … 0.99) | 0.01, std clamp |

```
dotnet add package Tsfm.Forecasting.T0
dotnet add package TorchSharp-cpu
./scripts/fetch-t0-checkpoint.sh            # both; or name one: t0-alpha | t0-beta
```

```csharp
using Tsfm.Forecasting.T0;

var t0 = T0Forecaster.Load("checkpoints/t0-beta", new Device("mps"));

// Same shape as TimesFM: variate-major series, known-future flags, target index.
double[,] q = t0.ForecastJoint(series, variates: 3, contextBars: 512, horizon: 24,
                               knownFuture: [false, true, true]);   // [step, quantile]
double[,] r = t0.Forecast(ohlcva, horizon: 4);                      // forward returns
```

Both forecasters implement `IQuantileForecaster`, so one call site drives either;
`Quantiles` names the columns, whose count differs per checkpoint. `samples/GroceryCovariates`
runs on either (`-- t0-beta`).

What differs from TimesFM:

- **Any context length.** The context is left-padded onto the 32-step patch grid and the
  padding masked, so nothing need fill whole patches. No ceiling is imposed; the reference
  evaluates at 8192.
- **Up to 1024 steps per pass.** Every horizon patch is decoded at once. The reference
  continues past 1024 by an autoregressive rollout over quantile paths, which is not ported.
- **Known-future series cannot read the targets.** Group attention lets a known-future row
  attend only to other known-future rows, so information flows from covariates into
  targets and never back. They are scaled by global rather than causal statistics.
- **Each checkpoint names its own scaler convention.** t0-beta's `config.json` carries
  `scaler_eps` and `scaler_eps_mode`; alpha's predates them, so their absence means alpha's
  values. Running beta under alpha's convention loads cleanly and silently degrades the
  forecast — which is why they are read rather than assumed.

## Choose your own backend

Neither model package propagates a native runtime, because the right one is platform-
and accelerator-specific. `TorchSharp-cpu` resolves it per RID; see
[TorchSharp's download guidance](https://github.com/dotnet/TorchSharp#download) for CUDA.

## Things that will bite you

**`DisposeScope` is mandatory.** TorchSharp has no refcounting. Every forward pass these
libraries perform is scoped; if you write your own, wrap it in
`using var _ = torch.NewDisposeScope()`. Without it the Metal allocator thrashes and you
will measure roughly a 10x slowdown that looks like a backend problem and is not.

**Cached tensors need `DetachFromDisposeScope()`, not `MoveToOuterDisposeScope()`.** The
latter hands ownership to the caller's scope, which frees it on exit — so a cache is
released after its first use.

**Metal is present in the "cpu" backend.** On Apple Silicon "cpu" means *not CUDA*. Probe
by placing a tensor rather than trusting the name.

**macOS needs Homebrew's libomp.** `libtorch_cpu.dylib` links
`/opt/homebrew/opt/libomp/lib/libomp.dylib` by absolute path, so the copy shipped inside
the NuGet package is never used. Without it, loading fails with a message claiming the
backend reference is missing — which it is not.

**No model has a key-value cache.** Attention is recomputed over the whole context
at every decode step, matching the references. A cache would change the arithmetic and
forfeit checkable parity. For Kronos this costs `horizon` full passes per rollout, so
long horizons are expensive; TimesFM and t0 emit their whole horizon in one pass and are
unaffected.

## Parity

Both ports are verified stage by stage against their references, on CPU and Metal,
rather than judged on final outputs — agreement at the end cannot distinguish a wrong
attention scale from a misplaced norm, since both merely shift the result.

**Kronos.** The tokenizer's embedding and first norm are bit-exact; relative error
thereafter is 1e-7 to 1e-6, ordinary float32 accumulation. It is *not* bit-identical and
does not claim to be: cross-implementation token disagreement extrapolates to near 1 in
15,000. Sampling differs deliberately — per-bar uniforms from a SplitMix64 stream seeded
by the bar's own timestamp, selected by inverse CDF, so a draw does not depend on how
bars were grouped into batches. The reference seeds one stream per batch and is not
batch-invariant. Distributionally identical; the stream is not.

**TimesFM.** Relative error is ~1e-7 through preprocessing and the layer stack, ~2e-6 at
the logits. Two faults the harness caught, neither of which throws and both of which
produce entirely plausible forecasts: `scaled_dot_product_attention` ignores `is_casual`
once `attn_mask` is supplied, so supplying a patch mask silently disabled causal masking
and let every position attend to its own future; and the reference leaves
`rescale_logits` false, making its logits `QK^T * sqrt(d)` rather than the conventional
`QK^T / sqrt(d)`.

**t0.** Every stage — scaling, attention masks, patch embedding, each layer, the
quantile head, the rescale — agrees with `tfc-t0` 0.6.0 to float32 rounding: masks and
scaling are bit-exact, layers within 6e-7 relative, forecasts within 5e-7. Each stage is
fed the reference's previous output, so a fault cannot hide behind an earlier one. The
end-to-end cases include the quickstart notebook's own examples: on the electricity-price
series, t0-beta's median misses the truth by 6.522 without covariates and 2.067 with them
(notebook: 6.528 without, recorded on different hardware), and the port reproduces
the reference to the same digits. Three details would each have produced plausible
forecasts if missed: rotary pairs are interleaved where TimesFM's are split in halves; the
QK norm precedes the rotation, where TimesFM's follows it; and a padding patch has no key
to attend to, so attention must return zero there, not NaN, or every row reading it is
poisoned.

```bash
./scripts/fetch-checkpoints.sh                       # Kronos
dotnet test tests/Tsfm.Forecasting.Kronos.Tests -c Release

./scripts/fetch-timesfm-checkpoint.sh                # TimesFM, non-commercial
./scripts/fetch-timesfm-reference.sh
python3 -m venv .venv && ./.venv/bin/pip install torch safetensors numpy huggingface_hub
./.venv/bin/python reference/dump_timesfm_parity.py
dotnet run --project tests/Tsfm.Forecasting.TimesFm.Parity -c Release

./scripts/fetch-t0-checkpoint.sh                     # t0, Apache-2.0
./.venv/bin/pip install "tfc-t0[torch]==0.6.0" pandas pyarrow
./.venv/bin/python reference/dump_t0_parity.py checkpoints/t0-alpha
./.venv/bin/python reference/dump_t0_parity.py checkpoints/t0-beta
dotnet run --project tests/Tsfm.Forecasting.T0.Parity -c Release -- t0-alpha t0-beta
```

## Attribution

**Kronos.** Model architecture, BSQ tokenizer and reference implementation by
[ShiYu](https://github.com/shiyu-coder/Kronos) (MIT). Published checkpoints are by
NeoQuasar (MIT), redistributed unmodified in the `Tsfm.Forecasting.Kronos.Weights.*`
packages, which record the revision each was built from.

**TimesFM.** Derived from
[google-research/timesfm](https://github.com/google-research/timesfm), Copyright 2026
Google LLC, Apache-2.0. No Google source files and no weights are redistributed here;
both are fetched at pinned revisions by the scripts above.

**t0.** Derived from [tfc-t0](https://github.com/theforecastingcompany/tfc-t0),
Copyright 2026 The Forecasting Company, Apache-2.0, which in turn builds on Datadog's
[Toto](https://github.com/DataDog/toto) and Amazon's
[Chronos-2](https://github.com/amazon-science/chronos-forecasting), both Apache-2.0. No
weights or upstream sources are redistributed; checkpoints are fetched at pinned revisions.

This project is an independent port and is not affiliated with or endorsed by any of them.
See NOTICE for the full attribution and the statement of changes.

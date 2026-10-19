"""Dump reference tensors so the .NET t0 port can be bisected stage by stage.

Runs the published tfc-t0 package (pinned below) against a local checkpoint and records
every stage boundary of one forward pass — scaling, patch embedding, selected layers, the
head, the rescale — plus end-to-end `predict` outputs. A single end-to-end comparison
cannot tell an interleaved-vs-halves RoPE from a misplaced norm; the first diverging stage
can.

The end-to-end cases include the t0 quickstart notebook's electricity-price and M4-hourly
examples, so the port reproduces the published sample, not just synthetic input.

    pip install "tfc-t0[torch]==0.6.0" pandas pyarrow
    python reference/dump_t0_parity.py checkpoints/t0-alpha
    python reference/dump_t0_parity.py checkpoints/t0-beta
"""

import importlib.metadata
import pathlib
import sys
import urllib.request

import numpy as np
import torch
from safetensors.torch import save_file

from t0 import T0Forecaster, TimeSeries
from t0.mask import build_attention_masks
from t0.rollout import prepare_rollout_buffer
from t0.types import round_up

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "0.6.0"

# Not a multiple of the patch: exercises left padding, and a horizon that ends inside its
# last patch, leaving known-future rows PAD there.
T, H = 150, 40
DATA = "https://autogluon.s3.amazonaws.com/datasets/timeseries"


def synthetic() -> tuple[torch.Tensor, torch.Tensor]:
    """Two jointly forecast targets with distinct scale and drift, plus two known-future
    covariates; one NaN in each kind so the missing-value path is exercised."""
    g = torch.Generator().manual_seed(20261019)
    t = torch.arange(T + H, dtype=torch.float32)
    ctx = torch.stack([
        100.0 + 5.0 * torch.sin(t[:T] / 7.0) + 0.02 * t[:T],
        -3.0 + 0.25 * torch.cos(t[:T] / 5.0),
    ]) + 0.05 * torch.randn(2, T, generator=g)
    fut = torch.stack([
        20.0 + 2.0 * torch.sin(t / 7.0 + 0.3),
        (t % 24 < 6).float(),
    ])
    ctx[0, 40] = float("nan")
    fut[0, 90] = float("nan")
    return ctx, fut


def stages(model: T0Forecaster, ctx: torch.Tensor, fut: torch.Tensor) -> dict[str, torch.Tensor]:
    p = model.patch_size
    series = TimeSeries.from_array(ctx[None], fut[None])
    buffer = prepare_rollout_buffer(series, H, T, p)
    window = buffer.time_slice(0, round_up(T, p) + round_up(H, p))
    out = {
        "buf_values": window.variates, "buf_mask": window.mask.float(),
        "buf_groups": window.group_ids.float(), "buf_types": window.variate_type.float(),
    }
    scaled, ls = model.scaler.scale_input(window)
    out |= {"scaled": scaled.variates, "loc": ls.loc, "scale": ls.scale}

    patch = model.patcher.patch
    values, mask, vtype, groups = (patch(f) for f in (scaled.variates, scaled.mask, scaled.variate_type, scaled.group_ids))
    h = model.patch_encoder(values, mask, vtype)
    out["patch_embedding"] = h
    time_mask, group_mask = build_attention_masks(groups, vtype, mask)
    out |= {"time_mask": time_mask.float(), "group_mask": group_mask.float()}
    for i, layer in enumerate(model.transformer.layers):
        h = layer(h, time_attn_mask=time_mask, group_attn_mask=group_mask)
        if i in (0, 1, 2, len(model.transformer.layers) - 1):
            out[f"layer{i}_output"] = h
    h = model.transformer.out_norm(h)
    out["transformer_output"] = h
    decoded = model.decoder(h)
    out["decoder_output"] = decoded
    q = model.head(decoded.reshape(*decoded.shape[:-1], p, model.head.n_quantiles))
    out["head_output"] = q
    out["rescaled"] = model.scaler.rescale_predictions(q, ls, p)
    return out


def fetch(name: str) -> pathlib.Path:
    cache = ROOT / "reference" / "data"
    cache.mkdir(exist_ok=True)
    path = cache / name.replace("/", "_")
    if not path.exists():
        urllib.request.urlretrieve(f"{DATA}/{name}", path)
    return path


def notebook_cases() -> dict[str, torch.Tensor]:
    """The quickstart notebook's inputs, exactly as it builds them."""
    import pandas as pd

    m4 = pd.read_csv(fetch("m4_hourly/train.csv"))
    sid = m4["item_id"].iloc[0]
    full = m4.loc[m4["item_id"] == sid, "target"].to_numpy(dtype=np.float32)
    train = pd.read_parquet(fetch("electricity_price/train.parquet"))
    test = pd.read_parquet(fetch("electricity_price/test.parquet"))
    cov_cols = [c for c in test.columns if c not in ("id", "timestamp", "target")]
    cov = np.concatenate(
        [train[cov_cols].to_numpy(dtype=np.float32)[-512:].T, test[cov_cols].to_numpy(dtype=np.float32).T], axis=1
    )
    return {
        "m4_ctx": torch.as_tensor(full[:512]),
        "m4_truth": torch.as_tensor(full[512:560]),
        "ep_ctx": torch.as_tensor(train["target"].to_numpy(dtype=np.float32)[-512:]),
        "ep_cov": torch.as_tensor(cov),
        "ep_truth": torch.as_tensor(test["target"].to_numpy(dtype=np.float32)),
    }


def main() -> None:
    installed = importlib.metadata.version("tfc-t0")
    if installed != EXPECTED_VERSION:
        print(f"warning: tfc-t0 {installed} installed; the port was verified against {EXPECTED_VERSION}")
    ckpt = pathlib.Path(sys.argv[1]).resolve()
    model = T0Forecaster.from_pretrained(str(ckpt)).eval()
    levels = list(model.config.quantile_levels)
    torch.set_grad_enabled(False)

    ctx, fut = synthetic()
    out = {"in_ctx": ctx, "in_fut": fut}
    out |= stages(model, ctx, fut)
    # End to end through predict, at the trained levels so no interpolation intervenes.
    out["predict_joint"] = model.predict(ctx[None], H, levels, future_covariates=fut[None]).quantiles[0]
    out["predict_targets_only"] = model.predict(ctx[None], H, levels).quantiles[0]
    # A univariate context that is already patch-aligned, with no padding at all.
    out["predict_aligned"] = model.predict(ctx[0, 64:128][None], 32, levels).quantiles[0]

    nb = notebook_cases()
    out |= nb
    out["m4_predict"] = model.predict(nb["m4_ctx"][None], 48, levels).quantiles[0]
    out["ep_predict"] = model.predict(nb["ep_ctx"][None], 24, levels).quantiles[0]
    out["ep_predict_cov"] = model.predict(
        nb["ep_ctx"][None], 24, levels, future_covariates=nb["ep_cov"][None]).quantiles[0]
    mid = levels.index(0.5)
    for name in ("ep_predict", "ep_predict_cov"):
        err = (out[name][:, mid] - nb["ep_truth"]).abs().mean().item()
        print(f"  {name:16s} |median-truth| = {err:.4f}")

    dest = ROOT / "reference" / f"{ckpt.name}-parity.safetensors"
    save_file({k: v.contiguous().float() for k, v in out.items()}, str(dest))
    print(f"  wrote {dest} ({len(out)} tensors)")


if __name__ == "__main__":
    main()

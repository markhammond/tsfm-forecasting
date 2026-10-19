"""Dump reference tensors so the .NET Timestral port can be bisected stage by stage.

Runs mistralai/timestral (pinned below) against a local checkpoint and records each stage
of one covariate-path block — packed rows, scaling, patch embedding, masks, positions,
selected layers, head, denormalisation, refine frame — then end-to-end `forecast` outputs
with the reference forecaster's defaults (symmetric averaging and non-negative clipping
on) and with both off.

The reference decodes a decoder token for every patch of the window and discards those
before the horizon; the port builds only the horizon's. They cannot influence the
horizon's, so decoder stages are recorded for the horizon tokens only.

    pip install git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff pandas pyarrow
    python reference/dump_timestral_parity.py checkpoints/Timestral-5M
"""

import math
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import save_file

from timestral import TimestralForecaster
from timestral.layers import TwoStreamMask
from timestral.masks import (
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from timestral.model import ROLE_KNOWN_COV, ROLE_TARGET
from timestral.scaler import denormalize_quantiles

ROOT = pathlib.Path(__file__).resolve().parent.parent
T, H = 150, 40          # not patch multiples: left padding, and a two-patch horizon for the refine frame
DATA = ROOT / "reference" / "data"


def synthetic() -> tuple[torch.Tensor, torch.Tensor]:
    """Two targets of distinct scale, a numeric known covariate and a 0/1 indicator, with
    one NaN in a target and one in the numeric covariate."""
    g = torch.Generator().manual_seed(20261019)
    t = torch.arange(T + H, dtype=torch.float32)
    ctx = torch.stack([
        100.0 + 5.0 * torch.sin(t[:T] / 7.0) + 0.02 * t[:T],
        -3.0 + 0.25 * torch.cos(t[:T] / 5.0),
    ]) + 0.05 * torch.randn(2, T, generator=g)
    fut = torch.stack([20.0 + 2.0 * torch.sin(t / 7.0 + 0.3), (t % 24 < 6).float()])
    ctx[0, 40] = float("nan")
    fut[0, 90] = float("nan")
    return ctx, fut


def stages(fc: TimestralForecaster, ctx: torch.Tensor, fut: torch.Tensor) -> dict[str, torch.Tensor]:
    model, c = fc.model, fc.config
    ps = c.patch_size
    inputs, mask = fc._as_context(ctx[None], None)
    known = fc._as_covariates(fut[None], 1, T + H, "known_covariates")
    rows, mask, role_ids, kf, kfm, _ = fc._assemble_rows(inputs, mask, H, None, known)
    b, v, t = rows.shape
    context, mask = rows.reshape(b * v, t), mask.reshape(b * v, t)
    group_ids = torch.zeros(b * v, dtype=torch.long)
    h = math.ceil(H / ps)
    full = torch.cat([context, kf[:, : h * ps]], -1)
    full_obs = torch.cat([mask, kfm[:, : h * ps]], -1)
    out = {"buf_values": full, "buf_observed": full_obs.float(), "buf_groups": group_ids.float(),
           "buf_roles": role_ids.float(), "ctx_patches": torch.tensor([t // ps], dtype=torch.float32)}

    n = full.shape[1] // ps
    ll = n - h
    known_rows = role_ids == ROLE_KNOWN_COV
    scaled, loc, scale = model.scale(full, full_obs)
    scaled, loc, scale = model._known_cov_scaling(full, full_obs, scaled, loc, scale, known_rows)
    out |= {"scaled": scaled, "loc": loc, "scale": scale}
    x_enc = model.representation(scaled, full_obs)
    out["patch_embedding"] = x_enc

    obs = full_obs.unflatten(-1, (n, ps)).any(-1)
    key_valid = compute_enc_key_validity(obs, c.attend_missing_patches)
    s_rows = torch.full((b * v,), ll, dtype=torch.long)
    time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, bidirectional_rows=known_rows)
    variate_mask = build_two_stream_variate_mask(
        group_ids, key_valid, n, target_rows=role_ids == ROLE_TARGET, bidirectional_rows=known_rows)
    uniq, inv = torch.unique(group_ids, return_inverse=True)
    first = group_first_observed_patch(obs, inv, uniq.numel())[inv]
    positions = build_two_stream_positions(n, first)

    # Keep [Enc ; Dec(horizon)] — the slots the port builds.
    keep = torch.cat([torch.arange(n), n + torch.arange(ll, n)])
    out["time_mask"] = time_mask[:, :, keep][:, :, :, keep].float()
    out["variate_mask"] = variate_mask[keep].float()
    out["positions"] = positions[:, keep].float()

    x = torch.cat([x_enc, model.representation.forecast_tokens(b * v, n)], 1)
    two = TwoStreamMask(mask=time_mask, enc_len=n)
    tf = model.transformer
    cos, sin = tf.rope(x, positions)
    layers = tf.layers
    for i, layer in enumerate(layers):
        x = layer(x, two, variate_mask, cos, sin)
        if i in (0, 3, len(layers) - 1):
            out[f"layer{i}_output"] = x[:, keep]
    x = tf.out_norm(x)
    out["transformer_output"] = x[:, keep]
    preds = model.head(x[:, n + ll :])
    out["head_output"] = preds
    anchor_loc = loc.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
    anchor_scale = scale.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
    raw = denormalize_quantiles(preds, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), c.use_asinh)
    out["block_output"] = raw
    out["refined"] = fc._refine_block(raw, context, mask, role_ids == ROLE_TARGET)
    return out


def notebook_cases() -> dict[str, torch.Tensor]:
    import pandas as pd

    m4 = pd.read_csv(DATA / "m4_hourly_train.csv")
    sid = m4["item_id"].iloc[0]
    full = m4.loc[m4["item_id"] == sid, "target"].to_numpy(dtype=np.float32)
    train = pd.read_parquet(DATA / "electricity_price_train.parquet")
    test = pd.read_parquet(DATA / "electricity_price_test.parquet")
    cols = [c for c in test.columns if c not in ("id", "timestamp", "target")]
    cov = np.concatenate([train[cols].to_numpy(dtype=np.float32)[-512:].T, test[cols].to_numpy(dtype=np.float32).T], 1)
    return {
        "m4_ctx": torch.as_tensor(full[:512]),
        "m4_truth": torch.as_tensor(full[512:560]),
        "ep_ctx": torch.as_tensor(train["target"].to_numpy(dtype=np.float32)[-512:]),
        "ep_cov": torch.as_tensor(cov),
        "ep_truth": torch.as_tensor(test["target"].to_numpy(dtype=np.float32)),
    }


def main() -> None:
    ckpt = pathlib.Path(sys.argv[1]).resolve()
    fc = TimestralForecaster.from_pretrained(str(ckpt))
    torch.set_grad_enabled(False)

    ctx, fut = synthetic()
    out = {"in_ctx": ctx, "in_fut": fut}
    out |= stages(fc, ctx, fut)

    def predict(c, h, known=None, raw=False):
        f = fc.forecast(c[None], h, known_covariates=None if known is None else known[None],
                        symmetric_averaging=not raw, clip_nonnegative=not raw)
        return f.quantiles[0]                       # (V, h, Q)

    out["predict_joint"] = predict(ctx, H, fut)
    out["predict_joint_raw"] = predict(ctx, H, fut, raw=True)
    out["predict_targets_only"] = predict(ctx, H)
    out["predict_aligned"] = predict(ctx[0, 64:128][None], 32)[0]

    nb = notebook_cases()
    out |= nb
    out["m4_predict"] = predict(nb["m4_ctx"][None], 48)[0]
    out["ep_predict"] = predict(nb["ep_ctx"][None], 24)[0]
    out["ep_predict_cov"] = predict(nb["ep_ctx"][None], 24, nb["ep_cov"])[0]
    mid = fc.quantile_levels.index(0.5)
    for name in ("ep_predict", "ep_predict_cov"):
        err = (out[name][:, mid] - nb["ep_truth"]).abs().mean().item()
        print(f"  {name:16s} |median-truth| = {err:.4f}")

    dest = ROOT / "reference" / f"{ckpt.name}-parity.safetensors"
    save_file({k: v.contiguous().float() for k, v in out.items()}, str(dest))
    print(f"  wrote {dest} ({len(out)} tensors)")


if __name__ == "__main__":
    main()

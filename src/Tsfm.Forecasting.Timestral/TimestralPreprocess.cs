using static TorchSharp.torch;

namespace Tsfm.Forecasting.Timestral;

/// <summary>Role of a row. Values match the reference's role ids.</summary>
public static class Role
{
    public const long Target = 0, PastCovariate = 1, KnownCovariate = 2;
}

/// <summary>
/// Scaling, attention masks and positions around the transformer. Every per-step array is
/// <c>(rows, T)</c>, with <c>T</c> a whole number of patches; an observation mask is true
/// where a value was observed.
/// </summary>
public static class TimestralPreprocess
{
    /// <summary>
    /// Causal per-patch location and scale, broadcast over each patch's steps: each patch
    /// takes the running mean and deviation of every observed value up to its last step.
    /// Computed in float64 on the CPU, as the reference does — MPS has no float64.
    /// </summary>
    public static (Tensor Loc, Tensor Scale) CausalPatchStats(Tensor values, Tensor observed, int patch, double minimumScale)
    {
        using var scope = NewDisposeScope();
        var device = values.device;
        var x = values.to(ScalarType.Float64, CPU);
        var w = observed.to(ScalarType.Float64, CPU);

        var den = w.cumsum(-1).clamp_min(1.0);
        var means = (w * x).cumsum(-1) / den;
        var shifted = cat([zeros_like(means.narrow(1, 0, 1)), means.narrow(1, 0, means.shape[1] - 1)], -1);
        var m2 = ((x - shifted) * (x - means) * w).cumsum(-1);
        var scales = (m2 / (den - 1.0).clamp_min(1.0)).sqrt().clamp_min(minimumScale);

        var (r, n) = (values.shape[0], values.shape[1] / patch);
        Tensor PerPatch(Tensor t) => t.to(ScalarType.Float32).reshape(r, n, patch).select(-1, patch - 1)
            .repeat_interleave(patch, -1).to(device);
        return (PerPatch(means).MoveToOuterDisposeScope(), PerPatch(scales).MoveToOuterDisposeScope());
    }

    /// <summary>float32 machine epsilon, the reference's floor on a divisor.</summary>
    private const double Float32Eps = 1.1920928955078125e-07;

    /// <summary><c>asinh((x - loc) / scale)</c> where observed, 0 elsewhere.</summary>
    public static Tensor Affine(Tensor x, Tensor loc, Tensor scale, Tensor observed, bool asinh)
    {
        using var s = NewDisposeScope();
        var scaled = ((x - loc) / scale.clamp_min(Float32Eps)).masked_fill(~observed, 0.0);
        if (asinh) scaled = where(observed, scaled.asinh(), scaled);
        return scaled.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Non-causal per-row location and scale for known-future rows, over context and
    /// horizon alike. A row observed only at 0 and 1 is an indicator: it keeps loc 0,
    /// scale 1, and skips the asinh.
    /// </summary>
    public static (Tensor Loc, Tensor Scale, Tensor IsBinary) InstanceStats(Tensor x, Tensor observed, double minimumScale)
    {
        using var s = NewDisposeScope();
        var m = observed.to(x.dtype);
        var n = m.sum(-1, keepdim: true).clamp_min(1.0);
        var loc = (x * m).sum(-1, keepdim: true) / n;
        var variance = ((x - loc) * m).pow(2).sum(-1, keepdim: true) / n;
        var scale = variance.sqrt().clamp_min(minimumScale);
        var zeroOne = (x.eq(0.0) | x.eq(1.0) | ~observed).all(-1, keepdim: true);
        var binary = zeroOne & observed.any(-1, keepdim: true);
        loc = where(binary, zeros_like(loc), loc);
        scale = where(binary, ones_like(scale), scale);
        return (loc.MoveToOuterDisposeScope(), scale.MoveToOuterDisposeScope(), binary.MoveToOuterDisposeScope());
    }

    /// <summary>
    /// Scaled input for every row: causal for targets and past covariates, instance
    /// statistics for known-future rows.
    /// </summary>
    public static (Tensor Scaled, Tensor Loc, Tensor Scale) Scale(
        Tensor x, Tensor observed, Tensor roles, TimestralConfig c)
    {
        using var s = NewDisposeScope();
        var (loc, scale) = CausalPatchStats(x, observed, c.PatchSize, c.MinimumScale);
        var scaled = Affine(x, loc, scale, observed, c.UseAsinh);

        var known = roles.eq(Role.KnownCovariate);
        if (known.any().item<bool>())
        {
            var (kl, ks, binary) = InstanceStats(x, observed, c.MinimumScale);
            var ksc = Affine(x, kl, ks, observed, asinh: false);
            if (c.UseAsinh) ksc = where(observed & ~binary, ksc.asinh(), ksc);
            var sel = known.unsqueeze(1);
            scaled = where(sel, ksc, scaled);
            loc = where(sel, kl.expand_as(loc), loc);
            scale = where(sel, ks.expand_as(scale), scale);
        }
        return (scaled.MoveToOuterDisposeScope(), loc.MoveToOuterDisposeScope(), scale.MoveToOuterDisposeScope());
    }

    /// <summary>Encoder key validity <c>(rows, W)</c>: a patch from the series' first
    /// observed one on, observed or not. Leading unobserved patches are padding.</summary>
    public static Tensor KeyValid(Tensor observedPerPatch)
        => observedPerPatch.to(ScalarType.Int32).cumsum(-1).gt(0);

    /// <summary>First patch any row of each row's group observes, per row; 0 if none.</summary>
    public static Tensor FirstObserved(Tensor observedPerPatch, Tensor groups)
    {
        using var s = NewDisposeScope();
        var w = observedPerPatch.shape[1];
        var groupCount = groups.max().item<long>() + 1;
        var count = zeros([groupCount, w], ScalarType.Float32, groups.device)
            .index_add_(0, groups, observedPerPatch.to(ScalarType.Float32), 1.0);
        return count.gt(0).to(ScalarType.Int32).argmax(-1).index_select(0, groups).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Time mask over <c>[Enc(W) ; Dec(h)]</c>, <c>(rows, 1, W+h, W+h)</c>, true attends.
    /// <list type="bullet">
    /// <item>Enc to Enc: causal over valid keys, the diagonal always open; full for
    /// known-future rows.</item>
    /// <item>Enc to Dec: never.</item>
    /// <item>Dec to Enc: every valid context key, <c>j &lt; contextPatches</c> — never the
    /// horizon's encoder tokens, which only known-future rows populate.</item>
    /// <item>Dec to Dec: causal.</item>
    /// </list>
    /// </summary>
    public static Tensor TimeMask(Tensor keyValid, Tensor knownRows, int contextPatches, int horizonPatches)
    {
        using var s = NewDisposeScope();
        var (r, w) = (keyValid.shape[0], keyValid.shape[1]);
        var h = horizonPatches;
        var dev = keyValid.device;
        var diagonal = TorchSharp.torch.eye(w, dtype: ScalarType.Bool, device: dev).unsqueeze(0);
        var keys = keyValid.unsqueeze(1);
        var causal = ones([w, w], ScalarType.Bool, dev).tril().unsqueeze(0);
        var encEnc = where(knownRows.view(r, 1, 1), keys.expand(r, w, w), causal & keys) | diagonal;
        var encDec = zeros([r, w, h], ScalarType.Bool, dev);
        var inContext = arange(w, ScalarType.Int64, dev).lt(contextPatches).view(1, 1, w);
        var decEnc = (inContext & keys).expand(r, h, w);
        var decDec = ones([h, h], ScalarType.Bool, dev).tril().unsqueeze(0).expand(r, h, h);
        var top = cat([encEnc, encDec], 2);
        var bottom = cat([decEnc, decDec], 2);
        return cat([top, bottom], 1).unsqueeze(1).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Variate mask <c>(W+h, 1, rows, rows)</c>: a row reads rows of its own group, on
    /// encoder slots only valid keys. Two firewalls: a covariate never reads a target, and a
    /// known-future row reads only known-future rows — else a covariate's view of the
    /// future could relay a target's own future back into it. The diagonal is always open.
    /// </summary>
    public static Tensor VariateMask(Tensor groups, Tensor keyValid, Tensor roles, int horizonPatches)
    {
        using var s = NewDisposeScope();
        var r = keyValid.shape[0];
        var dev = keyValid.device;
        var sameGroup = groups.unsqueeze(1).eq(groups.unsqueeze(0));
        var slotValid = cat([keyValid, ones([r, horizonPatches], ScalarType.Bool, dev)], 1).t();   // (S, R)
        var target = roles.eq(Role.Target);
        var known = roles.eq(Role.KnownCovariate);
        var forbid = (~target.unsqueeze(1) & target.unsqueeze(0)) | (known.unsqueeze(1) & ~known.unsqueeze(0));
        var mask = sameGroup.unsqueeze(0) & slotValid.unsqueeze(1) & ~forbid.unsqueeze(0);
        var diagonal = TorchSharp.torch.eye(r, dtype: ScalarType.Bool, device: dev).unsqueeze(0);
        return (mask | diagonal).unsqueeze(1).MoveToOuterDisposeScope();
    }

    /// <summary>Positions <c>(rows, W+h)</c>: encoder patch <c>p</c> and the decoder token for
    /// horizon patch <c>contextPatches + i</c> sit at their patch index less the group's first
    /// observed patch.</summary>
    public static Tensor Positions(Tensor firstObserved, int encPatches, int contextPatches, int horizonPatches)
    {
        var dev = firstObserved.device;
        var p = cat([arange(encPatches, ScalarType.Int64, dev), arange(contextPatches, contextPatches + horizonPatches,
            ScalarType.Int64, dev)]);
        return p.unsqueeze(0) - firstObserved.unsqueeze(1);
    }

    /// <summary><c>sinh(x) * scale + loc</c>, then ±inf (a sinh overflow) to the nearest
    /// finite value along the quantile axis, or 0 when the whole axis is infinite.</summary>
    public static Tensor Denormalize(Tensor preds, Tensor loc, Tensor scale, bool asinh)
    {
        using var s = NewDisposeScope();
        var x = asinh ? preds.sinh() : preds;
        return ClampNonFinite(x * scale + loc).MoveToOuterDisposeScope();
    }

    public static Tensor ClampNonFinite(Tensor v)
    {
        using var s = NewDisposeScope();
        var inf = float.PositiveInfinity;
        var maxFinite = where(v.isfinite(), v, full_like(v, -inf)).amax([-1L], keepdim: true);
        var minFinite = where(v.isfinite(), v, full_like(v, inf)).amin([-1L], keepdim: true);
        var c = where(v.eq(inf), maxFinite.expand_as(v), where(v.eq(-inf), minFinite.expand_as(v), v));
        return where(c.isinf(), zeros_like(c), c).MoveToOuterDisposeScope();
    }
}

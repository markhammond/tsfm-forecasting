using static TorchSharp.torch;

namespace Tsfm.Forecasting.T0;

/// <summary>Role of a row. Values match the type embedding indices the checkpoint learned.</summary>
public static class VariateType
{
    public const long Target = 0, Historical = 1, Future = 2;
}

/// <summary>Why a cell is not an observation. <c>Withheld</c> marks steps to predict.</summary>
public static class MaskType
{
    public const long Valid = 0, Pad = 1, Missing = 2, Censored = 3, Withheld = 4;
}

/// <summary>
/// The scaling and masking that surround the transformer. Every array is per time step,
/// <c>(rows, width)</c>, with the width already aligned to whole patches.
///
/// <para>Targets are standardised by causal running statistics — each step by the mean and
/// deviation of every valid step up to and including it — and known-future rows by their
/// global statistics over context and horizon alike.</para>
/// </summary>
public static class T0Preprocess
{
    /// <summary>Causal mean and deviation along time, one series per row.</summary>
    /// <remarks>Cumulative sums rather than a recurrence, exactly as the reference forms
    /// them: the checkpoint was trained on these numerics, and the two eps conventions
    /// disagree wherever a local deviation sits near eps.</remarks>
    public static (Tensor Loc, Tensor Scale) CausalStats(Tensor x, Tensor invalid, double eps, bool stdClamp)
    {
        using var scope = NewDisposeScope();
        var valid = (~invalid).to(x.dtype);
        var count = valid.cumsum(-1).clamp_min(1.0);
        var masked = where(invalid, zeros_like(x), x);
        var means = masked.cumsum(-1) / count;
        var shifted = cat([zeros_like(means.narrow(1, 0, 1)), means.narrow(1, 0, means.shape[1] - 1)], -1);
        var m2 = ((masked - shifted) * (masked - means) * valid).cumsum(-1).clamp_min(0.0);
        var variance = m2 / (count - 1.0).clamp_min(1.0);
        var scale = stdClamp ? variance.sqrt().clamp_min(eps) : (variance + eps).sqrt();
        return (means.MoveToOuterDisposeScope(), scale.MoveToOuterDisposeScope());
    }

    /// <summary>Global mean and deviation per row, broadcast back over time. The deviation
    /// is always clamped at eps, whatever the causal convention.</summary>
    public static (Tensor Loc, Tensor Scale) GlobalStats(Tensor x, Tensor invalid, double eps)
    {
        using var scope = NewDisposeScope();
        var valid = (~invalid).to(x.dtype);
        var count = RowSum(valid);
        var masked = where(invalid, zeros_like(x), x);
        var mean = RowSum(masked) / count.clamp_min(1.0);
        var squared = where(invalid, zeros_like(x), (x - mean) * (x - mean));
        var std = (RowSum(squared) / count.clamp_min(2.0)).sqrt().clamp_min(eps);
        return (mean.expand_as(x).MoveToOuterDisposeScope(), std.expand_as(x).MoveToOuterDisposeScope());
    }

    /// <summary>Sum over the last axis by a one-bucket scatter-add, the summation order the
    /// reference uses. A plain sum reduces pairwise and differs in the last bits.</summary>
    private static Tensor RowSum(Tensor x)
    {
        var index = zeros(x.shape, ScalarType.Int64, x.device);
        return zeros([x.shape[0], 1], x.dtype, x.device).scatter_add_(-1, index, x);
    }

    /// <summary>Standardise every row; returns the scaled values and the statistics used,
    /// which <see cref="Rescale"/> needs to invert the model's output.</summary>
    public static (Tensor Scaled, Tensor Loc, Tensor Scale) ScaleInput(
        Tensor x, Tensor mask, Tensor groupIds, Tensor variateType, T0Config cfg)
    {
        using var scope = NewDisposeScope();
        var invalid = mask.ne(MaskType.Valid) | x.isnan();
        if (!cfg.ScalerZeroNoValue) x = where(x.isnan(), zeros_like(x), x);

        var (causalLoc, causalScale) = CausalStats(x, invalid, cfg.ScalerEps, cfg.ScalerStdClamp);
        var (futureLoc, futureScale) = GlobalStats(x, invalid, cfg.ScalerEps);

        // Padding carries group -1 and takes neither statistic.
        var real = groupIds.ge(0);
        var isFuture = real & variateType.eq(VariateType.Future);
        var isCausal = real & (variateType.eq(VariateType.Target) | variateType.eq(VariateType.Historical));
        var loc = where(isFuture, futureLoc, where(isCausal, causalLoc, zeros_like(x)));
        var scale = where(isFuture, futureScale, where(isCausal, causalScale, ones_like(x)));

        var scaled = (x - loc) / scale;
        if (cfg.ScalerUseArcsinh) scaled = scaled.asinh();
        if (cfg.ScalerZeroNoValue)
        {
            // A cell holding no observation reads as zero whatever placeholder it carried;
            // a censored cell keeps its scaled bound.
            var noValue = mask.eq(MaskType.Pad) | mask.eq(MaskType.Missing) | mask.eq(MaskType.Withheld);
            scaled = where(noValue, zeros_like(scaled), scaled);
        }
        return (scaled.MoveToOuterDisposeScope(), loc.MoveToOuterDisposeScope(), scale.MoveToOuterDisposeScope());
    }

    /// <summary>Invert scaling on per-patch predictions <c>(rows, patches, patch, quantiles)</c>,
    /// using the statistics at each patch's last step.</summary>
    public static Tensor Rescale(Tensor predictions, Tensor loc, Tensor scale, int patchSize, bool arcsinh)
    {
        using var scope = NewDisposeScope();
        var (rows, patches) = (predictions.shape[0], predictions.shape[1]);
        var l = loc.reshape(rows, patches, patchSize).select(-1, patchSize - 1).unsqueeze(-1).unsqueeze(-1);
        var s = scale.reshape(rows, patches, patchSize).select(-1, patchSize - 1).unsqueeze(-1).unsqueeze(-1);
        var values = arcsinh ? predictions.sinh() : predictions;
        return (values * s + l).MoveToOuterDisposeScope();
    }

    /// <summary>Collapse patched metadata <c>(rows, patches, patch)</c> to each patch's first
    /// non-padding step, or -1 where the patch is all padding.</summary>
    public static Tensor ReducePatchMetadata(Tensor patched, Tensor patchedMask)
    {
        using var scope = NewDisposeScope();
        var isReal = patchedMask.ne(MaskType.Pad);
        var first = isReal.to(ScalarType.Int32).argmax(-1, keepdim: true);
        var reduced = patched.gather(-1, first).squeeze(-1);
        return where(isReal.any(-1), reduced, full_like(reduced, -1)).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// The two attention masks of a forward pass, true where attention is allowed.
    /// <list type="bullet">
    /// <item>Time <c>(rows, 1, patches, patches)</c>: causal for targets and historicals,
    /// bidirectional for known-future rows; a patch reads only its own series, and nothing
    /// reads an all-padding patch.</item>
    /// <item>Group <c>(patches, 1, rows, rows)</c>: rows sharing a group attend to one
    /// another, but a known-future row reads only known-future rows — information flows
    /// from covariates into targets and never back.</item>
    /// </list>
    /// </summary>
    public static (Tensor Time, Tensor Group) AttentionMasks(
        Tensor patchedGroupIds, Tensor patchedVariateType, Tensor patchedMask)
    {
        using var scope = NewDisposeScope();
        var groups = ReducePatchMetadata(patchedGroupIds, patchedMask);        // (V, P)
        var types = ReducePatchMetadata(patchedVariateType, patchedMask);
        var attendable = patchedMask.ne(MaskType.Pad).any(-1);
        var valid = groups.ge(0);
        var isFuture = types.eq(VariateType.Future);

        var sameSeries = groups.unsqueeze(2).eq(groups.unsqueeze(1)) & valid.unsqueeze(2) & valid.unsqueeze(1);
        var positions = arange(groups.shape[1], ScalarType.Int64, groups.device);
        var causal = positions.unsqueeze(1).ge(positions.unsqueeze(0));
        var futureQuery = isFuture.unsqueeze(2);
        var time = (futureQuery & sameSeries) | (~futureQuery & sameSeries & causal);
        time = time & attendable.unsqueeze(1);

        var (ids, ok, fut) = (groups.t(), valid.t(), isFuture.t());
        var group = ids.unsqueeze(2).eq(ids.unsqueeze(1)) & ok.unsqueeze(2) & ok.unsqueeze(1);
        group = group & ~(fut.unsqueeze(2) & ~fut.unsqueeze(1));

        return (time.unsqueeze(1).MoveToOuterDisposeScope(), group.unsqueeze(1).MoveToOuterDisposeScope());
    }
}

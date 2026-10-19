using static TorchSharp.torch;

namespace Tsfm.Forecasting.TimesFm;

/// <summary>
/// Quantile forecasts of forward return from OHLCVA bars, with the six channels supplied
/// as variates.
///
/// <para>Every variate is marked a target. That is not cosmetic: the preprocessing masks
/// future covariate slots only for target variates, so a channel left non-target is handed
/// its own future values. Marking close alone would leak tomorrow's high, low and volume
/// into today's forecast and flatter the model enormously.</para>
/// </summary>
/// <remarks>Inference is not thread safe.</remarks>
public sealed class TimesFmForecaster(TimesFmModel model, TimesFmConfig config, Device device)
    : IQuantileForecaster
{
    private const int CloseChannel = 3;    // open, high, low, close, volume, amount
    private const int Channels = 6;

    public TimesFmConfig Config => config;

    /// <inheritdoc/>
    public IReadOnlyList<double> Quantiles => config.Quantiles;

    public static TimesFmForecaster Load(string checkpointDir, Device device)
    {
        var cfg = TimesFmConfig.Load(Path.Combine(checkpointDir, "config.json"));
        return new TimesFmForecaster(TimesFmModel.Load(checkpointDir, device), cfg, device);
    }


    /// <summary>
    /// Joint forecast over several series, any of which may be known into the future.
    ///
    /// <para>Variates are attended jointly, so related series inform one another. They are
    /// normalised independently, so mixing scales — a price in the thousands beside a
    /// promotion flag in {0,1} — needs no manual scaling.</para>
    ///
    /// <para>There is no variate identity or ordering: the model sees anonymous series and
    /// infers their relationship from co-movement in the supplied window alone. Order
    /// therefore carries no meaning, and cannot be got wrong.</para>
    /// </summary>
    /// <param name="series">Variate-major <c>[variates x (contextBars + horizon)]</c>, so
    /// series <c>v</c> at step <c>t</c> is <c>series[v * (contextBars + horizon) + t]</c>.
    /// The future region of a variate not marked known is never read.</param>
    /// <param name="variates">Series supplied, at most <see cref="MaxVariates"/>.</param>
    /// <param name="contextBars">Observed steps. With the horizon, must be a multiple of
    /// the patch length.</param>
    /// <param name="horizon">Steps ahead to return, 1..<c>OutputPatchLen</c>.</param>
    /// <param name="knownFuture">Per variate: true where the future region of
    /// <paramref name="series"/> holds real values the model may read. Empty means none.
    /// A variate marked known is NOT a forecast target — marking a series you are
    /// forecasting would hand it its own future.</param>
    /// <param name="targetVariate">Which series the returned quantiles describe.</param>
    /// <returns><c>[horizon, numQuantiles]</c> in the units of the target series.</returns>
    public double[,] ForecastJoint(
        ReadOnlySpan<float> series, int variates, int contextBars, int horizon,
        ReadOnlySpan<bool> knownFuture = default, int targetVariate = 0)
    {
        var p = config.InputPatchLen;
        if (variates < 1 || variates > MaxVariates)
            throw new ArgumentOutOfRangeException(nameof(variates),
                $"variates must lie in 1..{MaxVariates}; got {variates}");
        if (series.Length % variates != 0)
            throw new ArgumentException(
                $"series length {series.Length} is not divisible by {variates} variates", nameof(series));
        var total = series.Length / variates;

        // The known future is gathered by rolling whole patches forward, so the series is
        // patched and the context must end on a patch boundary, leaving at least one patch
        // after it to roll into.
        if (total % p != 0)
            throw new ArgumentException(
                $"each series must hold a multiple of the patch length {p}; got {total}", nameof(series));
        if (contextBars % p != 0 || contextBars < p || contextBars >= total)
            throw new ArgumentException(
                $"contextBars ({contextBars}) must be a positive multiple of the patch length {p} " +
                $"and leave at least one patch of the {total} supplied", nameof(contextBars));
        if (horizon > total - contextBars)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"horizon {horizon} exceeds the {total - contextBars} steps supplied after the context");
        if (horizon < 1 || horizon > config.OutputPatchLen)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"horizon must lie in 1..{config.OutputPatchLen}");
        if (targetVariate < 0 || targetVariate >= variates)
            throw new ArgumentOutOfRangeException(nameof(targetVariate));
        if (!knownFuture.IsEmpty && knownFuture.Length != variates)
            throw new ArgumentException(
                $"knownFuture must hold {variates} flags or be empty", nameof(knownFuture));
        if (!knownFuture.IsEmpty && knownFuture[targetVariate])
            throw new ArgumentException(
                "the target variate cannot be known into the future", nameof(knownFuture));

        var n = total / p;
        var anchorPatch = contextBars / p - 1;      // last patch made entirely of observed steps
        using var scope = NewDisposeScope();
        using var _ = no_grad();

        var flat = zeros(variates * total, ScalarType.Float32);
        var dst = flat.data<float>();
        for (var i = 0; i < variates * total; i++) dst[i] = series[i];
        var values = flat.view(variates, n, p).unsqueeze(0).contiguous().to(device);

        // Mask the future region of every variate whose future is not supplied, and mark
        // as target exactly those variates. Only a target has its future slots withheld,
        // so a variate left unmarked would be handed values it should not see.
        var masks = zeros([1, variates, n, p], ScalarType.Bool, device);
        var isTarget = ones([1, variates, n], ScalarType.Bool, device);
        for (var v = 0; v < variates; v++)
        {
            var known = !knownFuture.IsEmpty && knownFuture[v];
            if (known) { isTarget[0, v] = false; continue; }
            for (var j = anchorPatch + 1; j < n; j++) masks[0, v, j] = true;
        }

        var logits = model.forward(values, masks, isTarget)
            .view(1, variates, n, config.OutputPatchLen, config.NumQuantiles);

        var (_, mu, sigma) = TimesFmPreprocess.RunningStats(
            nan_to_num(values, 0.0).clamp(-config.ValueClip, config.ValueClip), masks);
        var m = mu[0, targetVariate, anchorPatch].item<float>();
        var sd = sigma[0, targetVariate, anchorPatch].item<float>();

        var head = logits[0, targetVariate, anchorPatch].to(ScalarType.Float32).cpu();
        var acc = head.data<float>();

        var result = new double[horizon, config.NumQuantiles];
        for (var h = 0; h < horizon; h++)
        for (var q = 0; q < config.NumQuantiles; q++)
            result[h, q] = acc[h * config.NumQuantiles + q] * sd + m;
        return result;
    }

    /// <summary>Most series this checkpoint attends over jointly, as it declares. Read
    /// from the checkpoint rather than assumed: it sits in the same config as the layer
    /// and head counts, and another checkpoint may state a different figure.</summary>
    public int MaxVariates => config.MaxVariates;

    /// <summary>
    /// Forward-return quantiles for each step up to <paramref name="horizon"/>.
    /// </summary>
    /// <param name="ohlcva">Row-major <c>[L x 6]</c>: open, high, low, close, volume,
    /// amount. L must be a multiple of the patch length. Every channel is supplied as a
    /// variate and marked a target, so none is handed its own future values.</param>
    /// <param name="horizon">Steps ahead to return, 1..<c>OutputPatchLen</c>. All of them
    /// come from one forward pass, so a longer horizon costs nothing extra.</param>
    /// <returns>[horizon, numQuantiles] of cumulative return relative to the anchor close.</returns>
    public double[,] Forecast(ReadOnlySpan<float> ohlcva, int horizon)
    {
        var p = config.InputPatchLen;
        var bars = ohlcva.Length / Channels;
        if (bars % p != 0)
            throw new ArgumentException($"bar count {bars} must be a multiple of the patch length {p}");
        if (horizon < 1 || horizon > config.OutputPatchLen)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"horizon must lie in 1..{config.OutputPatchLen}");

        var n = bars / p;
        using var scope = NewDisposeScope();
        using var _ = no_grad();

        // (bars, 6) -> (1, 6, n, p): variate-major, then patched along time.
        var flat = zeros(bars * Channels, ScalarType.Float32);
        var dst = flat.data<float>();
        for (var i = 0; i < bars * Channels; i++) dst[i] = ohlcva[i];
        var values = flat.view(bars, Channels).t().reshape(1, Channels, n, p).contiguous().to(device);

        var masks = zeros([1, Channels, n, p], ScalarType.Bool, device);
        var isTarget = ones([1, Channels, n], ScalarType.Bool, device);

        var logits = model.forward(values, masks, isTarget)
            .view(1, Channels, n, config.OutputPatchLen, config.NumQuantiles);

        // The final patch carries the forecast; its running statistics undo the
        // normalisation the model produced its output in.
        var (_, mu, sigma) = TimesFmPreprocess.RunningStats(
            nan_to_num(values, 0.0).clamp(-config.ValueClip, config.ValueClip), masks);
        var m = mu[0, CloseChannel, n - 1].item<float>();
        var s = sigma[0, CloseChannel, n - 1].item<float>();

        var head = logits[0, CloseChannel, n - 1].to(ScalarType.Float32).cpu();
        var acc = head.data<float>();

        var anchorClose = ohlcva[(bars - 1) * Channels + CloseChannel];
        var result = new double[horizon, config.NumQuantiles];
        for (var h = 0; h < horizon; h++)
        for (var q = 0; q < config.NumQuantiles; q++)
        {
            var price = acc[h * config.NumQuantiles + q] * s + m;   // reverse revin
            result[h, q] = price / anchorClose - 1.0;
        }
        return result;
    }
}

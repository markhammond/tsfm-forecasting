using static TorchSharp.torch;

namespace Tsfm.Forecasting.T0;

/// <summary>
/// Quantile forecasts from a t0 checkpoint, in the same shape as
/// <c>TimesFmForecaster</c>: a joint forecast over related series, any of which may be
/// known into the future, and a forward-return forecast from OHLCVA bars.
///
/// <para>Unlike TimesFM, nothing need fill whole patches: a context of any length is
/// left-padded onto the patch grid, and the padding is masked out of attention.</para>
///
/// <para>Every series not marked known is a forecast target, and targets are forecast
/// jointly — so a channel cannot be handed its own future. Known-future series are
/// conditioned on but not forecast, and by construction cannot read the targets: their
/// group attention reads only other known-future rows.</para>
/// </summary>
/// <remarks>Inference is not thread safe.</remarks>
public sealed class T0Forecaster(T0Model model, T0Config config, Device device) : IQuantileForecaster
{
    /// <summary>Longest horizon decoded in one pass. The reference continues beyond it by an
    /// autoregressive rollout over quantile paths, which is not ported.</summary>
    public const int MaxHorizon = 1024;

    private const int CloseChannel = 3;    // open, high, low, close, volume, amount
    private const int Channels = 6;

    public T0Config Config => config;

    public T0Model Model => model;

    /// <inheritdoc/>
    public IReadOnlyList<double> Quantiles => config.Quantiles;

    public static T0Forecaster Load(string checkpointDir, Device device)
    {
        var net = T0Model.Load(checkpointDir, device);
        return new T0Forecaster(net, net.Config, device);
    }

    /// <summary>
    /// Joint forecast over several series, any of which may be known into the future.
    ///
    /// <para>Series are scaled independently — causally for targets, globally for
    /// known-future series — so a price in the thousands beside a {0,1} flag needs no
    /// manual scaling. There is no variate identity or ordering; the model infers each
    /// series' relevance from the supplied window alone.</para>
    /// </summary>
    /// <param name="series">Variate-major <c>[variates x total]</c>, so series <c>v</c> at
    /// step <c>t</c> is <c>series[v * total + t]</c> where <c>total = series.Length /
    /// variates</c>. Steps from <paramref name="contextBars"/> on are read only for known
    /// variates, and only the first <paramref name="horizon"/> of them. NaN is a missing
    /// observation.</param>
    /// <param name="variates">Series supplied.</param>
    /// <param name="contextBars">Observed steps, any positive number.</param>
    /// <param name="horizon">Steps ahead to return, 1..<see cref="MaxHorizon"/>.</param>
    /// <param name="knownFuture">Per variate: true where its future region holds real values
    /// the model may read. Empty means none. A variate marked known is NOT a forecast
    /// target.</param>
    /// <param name="targetVariate">Which series the returned quantiles describe.</param>
    /// <returns><c>[horizon, Quantiles.Count]</c> in the units of the target series.</returns>
    public double[,] ForecastJoint(
        ReadOnlySpan<float> series, int variates, int contextBars, int horizon,
        ReadOnlySpan<bool> knownFuture = default, int targetVariate = 0)
    {
        if (variates < 1)
            throw new ArgumentOutOfRangeException(nameof(variates), "at least one series is required");
        if (series.Length % variates != 0)
            throw new ArgumentException(
                $"series length {series.Length} is not divisible by {variates} variates", nameof(series));
        var total = series.Length / variates;
        ValidateHorizon(horizon);
        if (contextBars < 1 || contextBars > total)
            throw new ArgumentOutOfRangeException(nameof(contextBars),
                $"contextBars must lie in 1..{total}; got {contextBars}");
        if (targetVariate < 0 || targetVariate >= variates)
            throw new ArgumentOutOfRangeException(nameof(targetVariate));
        if (!knownFuture.IsEmpty && knownFuture.Length != variates)
            throw new ArgumentException(
                $"knownFuture must hold {variates} flags or be empty", nameof(knownFuture));
        if (!knownFuture.IsEmpty && knownFuture[targetVariate])
            throw new ArgumentException(
                "the target variate cannot be known into the future", nameof(knownFuture));
        var anyKnown = !knownFuture.IsEmpty && knownFuture.Contains(true);
        if (anyKnown && horizon > total - contextBars)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"horizon {horizon} exceeds the {total - contextBars} known steps supplied after the context");

        // Targets first, then known-future rows: the reference's row order. Attention is
        // indifferent to it, but summation order is not, and parity is checked bit-close.
        var order = new int[variates];
        var known = new bool[variates];
        var rows = 0;
        for (var v = 0; v < variates; v++)
            if (knownFuture.IsEmpty || !knownFuture[v]) order[rows++] = v;
        for (var v = 0; v < variates; v++)
            if (!knownFuture.IsEmpty && knownFuture[v]) { known[rows] = true; order[rows++] = v; }

        var layout = new Layout(config.PatchSize, variates, contextBars, horizon);
        var targetRow = Array.IndexOf(order, targetVariate);
        for (var r = 0; r < variates; r++)
        {
            var src = series.Slice(order[r] * total, total);
            layout.Row(r, src[..contextBars], known[r] ? src.Slice(contextBars, horizon) : default);
        }

        var head = Run(layout, targetRow);
        var result = new double[horizon, config.NumQuantiles];
        for (var h = 0; h < horizon; h++)
        for (var q = 0; q < config.NumQuantiles; q++)
            result[h, q] = head[h * config.NumQuantiles + q];
        return result;
    }

    /// <summary>
    /// Forward-return quantiles for each step up to <paramref name="horizon"/>.
    /// </summary>
    /// <param name="ohlcva">Row-major <c>[L x 6]</c>: open, high, low, close, volume,
    /// amount. Any L. Every channel is a target, forecast jointly, so none is handed its
    /// own future values.</param>
    /// <param name="horizon">Steps ahead to return, 1..<see cref="MaxHorizon"/>, all from
    /// one forward pass.</param>
    /// <returns>[horizon, numQuantiles] of cumulative return relative to the anchor close.</returns>
    public double[,] Forecast(ReadOnlySpan<float> ohlcva, int horizon)
    {
        if (ohlcva.Length == 0 || ohlcva.Length % Channels != 0)
            throw new ArgumentException($"ohlcva must hold whole bars of {Channels} channels", nameof(ohlcva));
        ValidateHorizon(horizon);
        var bars = ohlcva.Length / Channels;

        var layout = new Layout(config.PatchSize, Channels, bars, horizon);
        var column = new float[bars];
        for (var c = 0; c < Channels; c++)
        {
            for (var t = 0; t < bars; t++) column[t] = ohlcva[t * Channels + c];
            layout.Row(c, column, default);
        }

        var head = Run(layout, CloseChannel);
        var anchorClose = ohlcva[(bars - 1) * Channels + CloseChannel];
        var result = new double[horizon, config.NumQuantiles];
        for (var h = 0; h < horizon; h++)
        for (var q = 0; q < config.NumQuantiles; q++)
            result[h, q] = head[h * config.NumQuantiles + q] / anchorClose - 1.0;
        return result;
    }

    private static void ValidateHorizon(int horizon)
    {
        if (horizon < 1 || horizon > MaxHorizon)
            throw new ArgumentOutOfRangeException(nameof(horizon), $"horizon must lie in 1..{MaxHorizon}");
    }

    /// <summary>One forward pass; the target row's first <c>horizon</c> steps as
    /// <c>[horizon * quantiles]</c>, in data units.</summary>
    private float[] Run(Layout layout, int targetRow)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();

        long[] dims = [layout.Rows, layout.Width];
        var values = tensor(layout.Values, dims).to(device);
        var mask = tensor(layout.Mask, dims).to(device);
        var groups = tensor(layout.Groups, dims).to(device);
        var types = tensor(layout.Types, dims).to(device);

        var (scaled, loc, scale) = T0Preprocess.ScaleInput(values, mask, groups, types, config);
        var quantiles = model.Backbone(scaled, mask, groups, types);
        var data = T0Preprocess.Rescale(quantiles, loc, scale, config.PatchSize, config.ScalerUseArcsinh);

        // The output at patch i forecasts patch i + 1, so the horizon starts at the last
        // context patch.
        var p = config.PatchSize;
        var forecast = data[targetRow]
            .narrow(0, layout.ContextPatches - 1, layout.ForecastPatches)
            .reshape(layout.ForecastPatches * p, config.NumQuantiles)
            .narrow(0, 0, layout.Horizon);
        forecast = nan_to_num(forecast, 0.0, 0.0, 0.0);   // the reference zeroes non-finite output
        return forecast.cpu().contiguous().data<float>().ToArray();
    }

    /// <summary>
    /// The per-step input the model reads: the context left-padded onto the patch grid, then
    /// the horizon rounded up to whole patches. Over the horizon a target is withheld and a
    /// known-future row carries its values; padding is inert (value 0, group and role -1).
    /// </summary>
    private sealed class Layout
    {
        public readonly int Rows, Width, PadLeft, Context, Horizon, ContextPatches, ForecastPatches;
        public readonly float[] Values;
        public readonly long[] Mask, Groups, Types;

        public Layout(int patch, int rows, int context, int horizon)
        {
            (Rows, Context, Horizon) = (rows, context, horizon);
            PadLeft = (patch - context % patch) % patch;
            ForecastPatches = (horizon + patch - 1) / patch;
            ContextPatches = (PadLeft + context) / patch;
            Width = PadLeft + context + ForecastPatches * patch;
            Values = new float[rows * Width];
            Mask = new long[rows * Width];
            Groups = new long[rows * Width];
            Types = new long[rows * Width];
        }

        /// <param name="row">Row index.</param>
        /// <param name="context">Observed steps.</param>
        /// <param name="future">Known future values, or empty for a target.</param>
        public void Row(int row, ReadOnlySpan<float> context, ReadOnlySpan<float> future)
        {
            var isFuture = !future.IsEmpty;
            var o = row * Width;
            for (var i = 0; i < Width; i++)
            {
                var pad = i < PadLeft;
                Groups[o + i] = pad ? -1 : 0;
                Types[o + i] = pad ? -1 : isFuture ? VariateType.Future : VariateType.Target;
                if (pad) { Mask[o + i] = MaskType.Pad; continue; }

                var t = i - PadLeft;
                if (t < Context) Write(o + i, context[t]);
                else if (!isFuture) Mask[o + i] = MaskType.Withheld;
                else if (t - Context < Horizon) Write(o + i, future[t - Context]);
                else Mask[o + i] = MaskType.Pad;     // the horizon's last patch, beyond what was asked
            }
        }

        private void Write(int at, float value)
        {
            if (float.IsNaN(value)) { Mask[at] = MaskType.Missing; return; }
            Values[at] = value;
        }
    }
}

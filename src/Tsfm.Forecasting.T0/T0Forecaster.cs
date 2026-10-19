using System.Runtime.InteropServices;

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
/// conditioned on but not forecast, and by construction cannot read the targets: in
/// variate attention a target reads every series of its entry, but a known-future series
/// reads only known-future series, itself included.</para>
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
        ValidateHorizon(horizon);
        var total = ValidateJoint(series.Length, variates, contextBars, horizon, knownFuture, targetVariate, "");
        var layout = new Layout(config.PatchSize, contextBars, horizon, variates);
        int[] targets = [AddJoint(layout, series, total, variates, contextBars, knownFuture, targetVariate)];

        var result = new double[horizon, config.NumQuantiles];
        Run(layout, targets, [0], MemoryMarshal.CreateSpan(ref result[0, 0], result.Length));
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
        ValidateHorizon(horizon);
        ValidateBars(ohlcva.Length, "");
        var layout = new Layout(config.PatchSize, ohlcva.Length / Channels, horizon, Channels);
        int[] targets = [AddBars(layout, ohlcva)];

        var result = new double[horizon, config.NumQuantiles];
        var flat = MemoryMarshal.CreateSpan(ref result[0, 0], result.Length);
        Run(layout, targets, [0], flat);
        ToReturns(flat, ohlcva);
        return result;
    }

    /// <summary>Rows — series, summed over entries — attended in one forward pass. A batch
    /// is split into passes of at most this many, since group attention costs grow with the
    /// square of the rows in a pass. An entry wider than this still runs, alone.</summary>
    public int MaxRowsPerPass { get; set; } = 256;

    /// <inheritdoc cref="IQuantileForecaster.BatchOutputLength"/>
    public int BatchOutputLength(int entries, int horizon) => entries * horizon * config.NumQuantiles;

    /// <summary>
    /// <see cref="ForecastJoint"/> over many independent entries in as few passes as
    /// <see cref="MaxRowsPerPass"/> allows. Each entry is its own group, so entries never
    /// attend to one another; contexts may differ in length, each being left-padded to the
    /// longest in its pass and the padding masked.
    ///
    /// <para>Entries are ordered by context length before being split into passes, so a
    /// short entry is not padded out to a long one's width. Results are written in the
    /// caller's order regardless.</para>
    /// </summary>
    /// <param name="batch">Entries, each with the constraints of <see cref="ForecastJoint"/>.</param>
    /// <param name="horizon">Steps ahead, shared, 1..<see cref="MaxHorizon"/>.</param>
    /// <param name="quantiles">Receives <c>[entry][step][quantile]</c>; sized by
    /// <see cref="BatchOutputLength"/>.</param>
    public void ForecastJointBatch(ReadOnlySpan<JointSeries> batch, int horizon, Span<double> quantiles)
    {
        ValidateHorizon(horizon);
        ValidateDestination(quantiles.Length, batch.Length, horizon, nameof(quantiles));
        var totals = new int[batch.Length];
        var contexts = new int[batch.Length];
        var widths = new int[batch.Length];
        for (var i = 0; i < batch.Length; i++)
        {
            var e = batch[i];
            totals[i] = ValidateJoint(e.Series.Length, e.Variates, e.ContextBars, horizon,
                e.KnownFuture.Span, e.TargetVariate, $"batch[{i}]: ");
            (contexts[i], widths[i]) = (e.ContextBars, e.Variates);
        }

        foreach (var pass in Passes(contexts, widths))
        {
            var layout = new Layout(config.PatchSize, pass.Max(i => contexts[i]), horizon, pass.Sum(i => widths[i]));
            var targets = new int[pass.Length];
            for (var j = 0; j < pass.Length; j++)
            {
                var e = batch[pass[j]];
                targets[j] = AddJoint(layout, e.Series.Span, totals[pass[j]], e.Variates, e.ContextBars,
                    e.KnownFuture.Span, e.TargetVariate);
            }
            Run(layout, targets, pass, quantiles);
        }
    }

    /// <summary>
    /// <see cref="Forecast"/> over many OHLCVA windows — instruments, or anchors along one
    /// history — batched as <see cref="ForecastJointBatch"/> is. Windows may differ in
    /// length; slicing one history into overlapping windows copies nothing.
    /// </summary>
    /// <param name="ohlcva">One row-major <c>[L x 6]</c> window per entry.</param>
    /// <param name="horizon">Steps ahead, shared, 1..<see cref="MaxHorizon"/>.</param>
    /// <param name="returns">Receives <c>[entry][step][quantile]</c> of cumulative return
    /// relative to each window's anchor close; sized by <see cref="BatchOutputLength"/>.</param>
    public void ForecastBatch(ReadOnlySpan<ReadOnlyMemory<float>> ohlcva, int horizon, Span<double> returns)
    {
        ValidateHorizon(horizon);
        ValidateDestination(returns.Length, ohlcva.Length, horizon, nameof(returns));
        var contexts = new int[ohlcva.Length];
        for (var i = 0; i < ohlcva.Length; i++)
        {
            ValidateBars(ohlcva[i].Length, $"ohlcva[{i}]: ");
            contexts[i] = ohlcva[i].Length / Channels;
        }

        var widths = Enumerable.Repeat(Channels, ohlcva.Length).ToArray();
        foreach (var pass in Passes(contexts, widths))
        {
            var layout = new Layout(config.PatchSize, pass.Max(i => contexts[i]), horizon, pass.Length * Channels);
            var targets = new int[pass.Length];
            for (var j = 0; j < pass.Length; j++) targets[j] = AddBars(layout, ohlcva[pass[j]].Span);
            Run(layout, targets, pass, returns);
        }

        var block = horizon * config.NumQuantiles;
        for (var i = 0; i < ohlcva.Length; i++) ToReturns(returns.Slice(i * block, block), ohlcva[i].Span);
    }

    /// <summary>Entry indices grouped into passes: ordered by context so a pass pads to
    /// similar widths, and cut wherever the next entry would exceed the row budget.</summary>
    private List<int[]> Passes(int[] contexts, int[] rows)
    {
        var order = Enumerable.Range(0, contexts.Length).OrderBy(i => contexts[i]).ToArray();
        var passes = new List<int[]>();
        var (from, used) = (0, 0);
        for (var j = 0; j < order.Length; j++)
        {
            if (j > from && used + rows[order[j]] > MaxRowsPerPass)
            {
                passes.Add(order[from..j]);
                (from, used) = (j, 0);
            }
            used += rows[order[j]];
        }
        if (from < order.Length) passes.Add(order[from..]);
        return passes;
    }

    /// <summary>Add one joint entry as its own group: targets first, then known-future rows,
    /// the reference's row order — attention is indifferent to it, but summation order is
    /// not, and parity is checked bit-close. Returns the target's row.</summary>
    private static int AddJoint(Layout layout, ReadOnlySpan<float> series, int total, int variates,
        int contextBars, ReadOnlySpan<bool> knownFuture, int targetVariate)
    {
        var group = layout.NextGroup();
        var target = -1;
        for (var pass = 0; pass < 2; pass++)
        for (var v = 0; v < variates; v++)
        {
            var known = !knownFuture.IsEmpty && knownFuture[v];
            if (known != (pass == 1)) continue;
            var src = series.Slice(v * total, total);
            var row = layout.AddRow(group, src[..contextBars], known ? src.Slice(contextBars, layout.Horizon) : default);
            if (v == targetVariate) target = row;
        }
        return target;
    }

    /// <summary>Add one OHLCVA window as a group of six targets; returns the close's row.</summary>
    private static int AddBars(Layout layout, ReadOnlySpan<float> ohlcva)
    {
        var group = layout.NextGroup();
        var bars = ohlcva.Length / Channels;
        Span<float> column = bars <= 1024 ? stackalloc float[bars] : new float[bars];
        var close = -1;
        for (var c = 0; c < Channels; c++)
        {
            for (var t = 0; t < bars; t++) column[t] = ohlcva[t * Channels + c];
            var row = layout.AddRow(group, column, default);
            if (c == CloseChannel) close = row;
        }
        return close;
    }

    private static void ToReturns(Span<double> prices, ReadOnlySpan<float> ohlcva)
    {
        var anchorClose = ohlcva[^(Channels - CloseChannel)];
        for (var i = 0; i < prices.Length; i++) prices[i] = prices[i] / anchorClose - 1.0;
    }

    private static int ValidateJoint(int length, int variates, int contextBars, int horizon,
        ReadOnlySpan<bool> knownFuture, int targetVariate, string where)
    {
        if (variates < 1)
            throw new ArgumentOutOfRangeException(nameof(variates), $"{where}at least one series is required");
        if (length % variates != 0)
            throw new ArgumentException(
                $"{where}series length {length} is not divisible by {variates} variates", "series");
        var total = length / variates;
        if (contextBars < 1 || contextBars > total)
            throw new ArgumentOutOfRangeException(nameof(contextBars),
                $"{where}contextBars must lie in 1..{total}; got {contextBars}");
        if (targetVariate < 0 || targetVariate >= variates)
            throw new ArgumentOutOfRangeException(nameof(targetVariate), $"{where}no variate {targetVariate}");
        if (!knownFuture.IsEmpty && knownFuture.Length != variates)
            throw new ArgumentException(
                $"{where}knownFuture must hold {variates} flags or be empty", nameof(knownFuture));
        if (!knownFuture.IsEmpty && knownFuture[targetVariate])
            throw new ArgumentException(
                $"{where}the target variate cannot be known into the future", nameof(knownFuture));
        if (!knownFuture.IsEmpty && knownFuture.Contains(true) && horizon > total - contextBars)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"{where}horizon {horizon} exceeds the {total - contextBars} known steps supplied after the context");
        return total;
    }

    private static void ValidateBars(int length, string where)
    {
        if (length == 0 || length % Channels != 0)
            throw new ArgumentException($"{where}ohlcva must hold whole bars of {Channels} channels", "ohlcva");
    }

    private void ValidateDestination(int length, int entries, int horizon, string name)
    {
        if (length != BatchOutputLength(entries, horizon))
            throw new ArgumentException(
                $"{name} must hold exactly {BatchOutputLength(entries, horizon)} elements "
                + $"(entries x horizon x {config.NumQuantiles} quantiles); got {length}. Use {nameof(BatchOutputLength)}.",
                name);
    }

    private static void ValidateHorizon(int horizon)
    {
        if (horizon < 1 || horizon > MaxHorizon)
            throw new ArgumentOutOfRangeException(nameof(horizon), $"horizon must lie in 1..{MaxHorizon}");
    }

    /// <summary>One forward pass. Writes target row <c>j</c>'s first <c>horizon</c> steps,
    /// in data units, to <paramref name="destination"/> at slot <c>slots[j]</c>.</summary>
    private void Run(Layout layout, int[] targetRows, ReadOnlySpan<int> slots, Span<double> destination)
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
        var (p, q, h) = (config.PatchSize, config.NumQuantiles, layout.Horizon);
        var forecast = data.index_select(0, tensor(targetRows.Select(r => (long)r).ToArray()).to(device))
            .narrow(1, layout.ContextPatches - 1, layout.ForecastPatches)
            .reshape(targetRows.Length, layout.ForecastPatches * p, q)
            .narrow(1, 0, h);
        forecast = nan_to_num(forecast, 0.0, 0.0, 0.0);   // the reference zeroes non-finite output
        var host = forecast.cpu().contiguous().data<float>();

        var block = h * q;
        for (var j = 0; j < targetRows.Length; j++)
        {
            var dst = destination.Slice(slots[j] * block, block);
            for (var i = 0; i < block; i++) dst[i] = host[j * block + i];
        }
    }

    /// <summary>
    /// The per-step input the model reads, for one or many groups. Every row is right-aligned
    /// on the forecast start, which sits on a patch boundary: its context is left-padded to
    /// the longest in the pass, rounded up to whole patches, and the horizon rounded up after
    /// it. So a row's patch grid is the same alone or batched. Over the horizon a target is
    /// withheld and a known-future row carries its values; padding is inert (value 0, group
    /// and role -1).
    /// </summary>
    private sealed class Layout
    {
        public readonly int Rows, Width, ContextWidth, Horizon, ContextPatches, ForecastPatches;
        public readonly float[] Values;
        public readonly long[] Mask, Groups, Types;
        private int _row;
        private long _group;

        public Layout(int patch, int maxContext, int horizon, int rows)
        {
            (Rows, Horizon) = (rows, horizon);
            ContextPatches = (maxContext + patch - 1) / patch;
            ForecastPatches = (horizon + patch - 1) / patch;
            ContextWidth = ContextPatches * patch;
            Width = ContextWidth + ForecastPatches * patch;
            Values = new float[rows * Width];
            Mask = new long[rows * Width];
            Groups = new long[rows * Width];
            Types = new long[rows * Width];
        }

        public long NextGroup() => _group++;

        /// <param name="group">Rows sharing a group attend to one another.</param>
        /// <param name="context">Observed steps.</param>
        /// <param name="future">Known future values, or empty for a target.</param>
        /// <returns>The row written.</returns>
        public int AddRow(long group, ReadOnlySpan<float> context, ReadOnlySpan<float> future)
        {
            var row = _row++;
            var isFuture = !future.IsEmpty;
            var pad = ContextWidth - context.Length;
            var o = row * Width;
            for (var i = 0; i < Width; i++)
            {
                if (i < pad)
                {
                    (Mask[o + i], Groups[o + i], Types[o + i]) = (MaskType.Pad, -1, -1);
                    continue;
                }
                Groups[o + i] = group;
                Types[o + i] = isFuture ? VariateType.Future : VariateType.Target;

                var t = i - pad;
                var f = t - context.Length;
                if (f < 0) Write(o + i, context[t]);
                else if (!isFuture) Mask[o + i] = MaskType.Withheld;
                else if (f < Horizon) Write(o + i, future[f]);
                else Mask[o + i] = MaskType.Pad;     // the horizon's last patch, beyond what was asked
            }
            return row;
        }

        private void Write(int at, float value)
        {
            if (float.IsNaN(value)) { Mask[at] = MaskType.Missing; return; }
            Values[at] = value;
        }
    }
}

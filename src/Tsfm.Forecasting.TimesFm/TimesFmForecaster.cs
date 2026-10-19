using System.Runtime.InteropServices;

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
        var total = ValidateJoint(series.Length, variates, contextBars, horizon, knownFuture, targetVariate, "");
        var known = new bool[variates];
        if (!knownFuture.IsEmpty) knownFuture.CopyTo(known);

        var result = new double[horizon, config.NumQuantiles];
        JointCore(1, variates, total, contextBars, horizon, series.ToArray(), known, [targetVariate], [0],
            MemoryMarshal.CreateSpan(ref result[0, 0], result.Length));
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
        ValidateBars(ohlcva.Length, horizon, "");
        var result = new double[horizon, config.NumQuantiles];
        BarsCore(1, ohlcva.Length / Channels, horizon, ohlcva.ToArray(), [0],
            MemoryMarshal.CreateSpan(ref result[0, 0], result.Length));
        return result;
    }

    /// <summary>Entries stacked into one forward pass, at most. Memory grows linearly with
    /// it; variate attention never crosses entries.</summary>
    public int MaxBatchPerPass { get; set; } = 64;

    /// <inheritdoc cref="IQuantileForecaster.BatchOutputLength"/>
    public int BatchOutputLength(int entries, int horizon) => entries * horizon * config.NumQuantiles;

    /// <summary>
    /// <see cref="ForecastJoint"/> over many entries. TimesFM has no padding of variates or
    /// context, so only entries of one shape — variates, total length and context — share a
    /// pass; a batch of mixed shapes runs as one pass per shape. Backtest anchors along one
    /// history are a single shape, and batch fully.
    /// </summary>
    /// <param name="batch">Entries, each with the constraints of <see cref="ForecastJoint"/>.</param>
    /// <param name="horizon">Steps ahead, shared, 1..<c>OutputPatchLen</c>.</param>
    /// <param name="quantiles">Receives <c>[entry][step][quantile]</c>; sized by
    /// <see cref="BatchOutputLength"/>.</param>
    public void ForecastJointBatch(ReadOnlySpan<JointSeries> batch, int horizon, Span<double> quantiles)
    {
        ValidateDestination(quantiles.Length, batch.Length, horizon, nameof(quantiles));
        var shapes = new (int Variates, int Total, int Context)[batch.Length];
        for (var i = 0; i < batch.Length; i++)
        {
            var e = batch[i];
            var total = ValidateJoint(e.Series.Length, e.Variates, e.ContextBars, horizon,
                e.KnownFuture.Span, e.TargetVariate, $"batch[{i}]: ");
            shapes[i] = (e.Variates, total, e.ContextBars);
        }

        foreach (var pass in Passes(shapes))
        {
            var (v, total, context) = shapes[pass[0]];
            var values = new float[pass.Length * v * total];
            var known = new bool[pass.Length * v];
            var targets = new int[pass.Length];
            for (var j = 0; j < pass.Length; j++)
            {
                var e = batch[pass[j]];
                e.Series.Span.CopyTo(values.AsSpan(j * v * total, v * total));
                if (!e.KnownFuture.IsEmpty) e.KnownFuture.Span.CopyTo(known.AsSpan(j * v, v));
                targets[j] = e.TargetVariate;
            }
            JointCore(pass.Length, v, total, context, horizon, values, known, targets, pass, quantiles);
        }
    }

    /// <summary><see cref="Forecast"/> over many OHLCVA windows, one pass per bar count.</summary>
    /// <param name="ohlcva">One row-major <c>[L x 6]</c> window per entry, L a multiple of
    /// the patch length.</param>
    /// <param name="horizon">Steps ahead, shared, 1..<c>OutputPatchLen</c>.</param>
    /// <param name="returns">Receives <c>[entry][step][quantile]</c> of cumulative return;
    /// sized by <see cref="BatchOutputLength"/>.</param>
    public void ForecastBatch(ReadOnlySpan<ReadOnlyMemory<float>> ohlcva, int horizon, Span<double> returns)
    {
        ValidateDestination(returns.Length, ohlcva.Length, horizon, nameof(returns));
        var shapes = new (int Bars, int, int)[ohlcva.Length];
        for (var i = 0; i < ohlcva.Length; i++)
        {
            ValidateBars(ohlcva[i].Length, horizon, $"ohlcva[{i}]: ");
            shapes[i] = (ohlcva[i].Length / Channels, 0, 0);
        }

        foreach (var pass in Passes(shapes))
        {
            var bars = shapes[pass[0]].Bars;
            var values = new float[pass.Length * bars * Channels];
            for (var j = 0; j < pass.Length; j++)
                ohlcva[pass[j]].Span.CopyTo(values.AsSpan(j * bars * Channels, bars * Channels));
            BarsCore(pass.Length, bars, horizon, values, pass, returns);
        }
    }

    /// <summary>Entry indices grouped by shape, then cut at <see cref="MaxBatchPerPass"/>.</summary>
    private IEnumerable<int[]> Passes<TShape>(TShape[] shapes) where TShape : notnull
        => Enumerable.Range(0, shapes.Length)
            .GroupBy(i => shapes[i])
            .SelectMany(g => g.Chunk(Math.Max(1, MaxBatchPerPass)));

    /// <summary>
    /// One pass over <paramref name="count"/> entries of one shape, stacked on the batch
    /// axis. Writes entry <c>j</c> to <paramref name="destination"/> at slot <c>slots[j]</c>.
    /// </summary>
    private void JointCore(int count, int variates, int total, int contextBars, int horizon,
        float[] series, bool[] known, int[] targets, ReadOnlySpan<int> slots, Span<double> destination)
    {
        var p = config.InputPatchLen;
        var n = total / p;
        var anchorPatch = contextBars / p - 1;      // last patch made entirely of observed steps
        using var scope = NewDisposeScope();
        using var _ = no_grad();

        var values = tensor(series, [count, variates, n, p]).to(device);

        // Mask the future region of every variate whose future is not supplied, and mark
        // as target exactly those variates. Only a target has its future slots withheld,
        // so a variate left unmarked would be handed values it should not see.
        var masks = zeros([count, variates, n, p], ScalarType.Bool, device);
        var isTarget = ones([count, variates, n], ScalarType.Bool, device);
        for (var b = 0; b < count; b++)
        for (var v = 0; v < variates; v++)
        {
            if (known[b * variates + v]) { isTarget[b, v] = false; continue; }
            if (anchorPatch + 1 < n) masks[b, v, TensorIndex.Slice(anchorPatch + 1)] = true;
        }

        var logits = model.forward(values, masks, isTarget)
            .view(count, variates, n, config.OutputPatchLen, config.NumQuantiles);
        var (_, mu, sigma) = TimesFmPreprocess.RunningStats(
            nan_to_num(values, 0.0).clamp(-config.ValueClip, config.ValueClip), masks);

        Emit(count, variates, anchorPatch, targets, logits, mu, sigma, horizon, slots, destination, null);
    }

    /// <summary>One pass over <paramref name="count"/> OHLCVA windows of one length.</summary>
    private void BarsCore(int count, int bars, int horizon, float[] ohlcva, ReadOnlySpan<int> slots,
        Span<double> destination)
    {
        var p = config.InputPatchLen;
        var n = bars / p;
        using var scope = NewDisposeScope();
        using var _ = no_grad();

        // (count, bars, 6) -> (count, 6, n, p): variate-major, then patched along time.
        var values = tensor(ohlcva, [count, bars, Channels])
            .permute(0, 2, 1).reshape(count, Channels, n, p).contiguous().to(device);

        var masks = zeros([count, Channels, n, p], ScalarType.Bool, device);
        var isTarget = ones([count, Channels, n], ScalarType.Bool, device);

        var logits = model.forward(values, masks, isTarget)
            .view(count, Channels, n, config.OutputPatchLen, config.NumQuantiles);

        // The final patch carries the forecast; its running statistics undo the
        // normalisation the model produced its output in.
        var (_, mu, sigma) = TimesFmPreprocess.RunningStats(
            nan_to_num(values, 0.0).clamp(-config.ValueClip, config.ValueClip), masks);

        var anchors = new float[count];
        for (var b = 0; b < count; b++) anchors[b] = ohlcva[(b * bars + bars - 1) * Channels + CloseChannel];
        Emit(count, Channels, n - 1, Enumerable.Repeat(CloseChannel, count).ToArray(), logits, mu, sigma,
            horizon, slots, destination, anchors);
    }

    /// <summary>Reverse RevIN on each entry's target at the anchor patch and write it out,
    /// as returns on <paramref name="anchorClose"/> when given.</summary>
    private void Emit(int count, int variates, int anchorPatch, int[] targets, Tensor logits, Tensor mu,
        Tensor sigma, int horizon, ReadOnlySpan<int> slots, Span<double> destination, float[]? anchorClose)
    {
        var rows = tensor(targets.Select((t, b) => (long)(b * variates + t)).ToArray()).to(device);
        var q = config.NumQuantiles;
        var head = logits.select(2, anchorPatch).reshape(count * variates, config.OutputPatchLen, q)
            .index_select(0, rows).to(ScalarType.Float32).cpu().contiguous().data<float>();
        var m = mu.select(2, anchorPatch).reshape(count * variates).index_select(0, rows).cpu().data<float>();
        var sd = sigma.select(2, anchorPatch).reshape(count * variates).index_select(0, rows).cpu().data<float>();

        var block = horizon * q;
        var stride = config.OutputPatchLen * q;
        for (var b = 0; b < count; b++)
        {
            var dst = destination.Slice(slots[b] * block, block);
            for (var h = 0; h < horizon; h++)
            for (var k = 0; k < q; k++)
            {
                var value = head[b * stride + h * q + k] * sd[b] + m[b];      // reverse revin
                dst[h * q + k] = anchorClose is null ? value : value / anchorClose[b] - 1.0;
            }
        }
    }

    private int ValidateJoint(int length, int variates, int contextBars, int horizon,
        ReadOnlySpan<bool> knownFuture, int targetVariate, string where)
    {
        var p = config.InputPatchLen;
        if (variates < 1 || variates > MaxVariates)
            throw new ArgumentOutOfRangeException(nameof(variates),
                $"{where}variates must lie in 1..{MaxVariates}; got {variates}");
        if (length % variates != 0)
            throw new ArgumentException(
                $"{where}series length {length} is not divisible by {variates} variates", "series");
        var total = length / variates;

        // The known future is gathered by rolling whole patches forward, so the series is
        // patched and the context must end on a patch boundary, leaving at least one patch
        // after it to roll into.
        if (total % p != 0)
            throw new ArgumentException(
                $"{where}each series must hold a multiple of the patch length {p}; got {total}", "series");
        if (contextBars % p != 0 || contextBars < p || contextBars >= total)
            throw new ArgumentException(
                $"{where}contextBars ({contextBars}) must be a positive multiple of the patch length {p} " +
                $"and leave at least one patch of the {total} supplied", nameof(contextBars));
        if (horizon > total - contextBars)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"{where}horizon {horizon} exceeds the {total - contextBars} steps supplied after the context");
        if (horizon < 1 || horizon > config.OutputPatchLen)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"{where}horizon must lie in 1..{config.OutputPatchLen}");
        if (targetVariate < 0 || targetVariate >= variates)
            throw new ArgumentOutOfRangeException(nameof(targetVariate), $"{where}no variate {targetVariate}");
        if (!knownFuture.IsEmpty && knownFuture.Length != variates)
            throw new ArgumentException(
                $"{where}knownFuture must hold {variates} flags or be empty", nameof(knownFuture));
        if (!knownFuture.IsEmpty && knownFuture[targetVariate])
            throw new ArgumentException(
                $"{where}the target variate cannot be known into the future", nameof(knownFuture));
        return total;
    }

    private void ValidateBars(int length, int horizon, string where)
    {
        var p = config.InputPatchLen;
        var bars = length / Channels;
        if (length == 0 || length % Channels != 0 || bars % p != 0)
            throw new ArgumentException(
                $"{where}ohlcva must hold whole bars, a multiple of the patch length {p}; got {length} values", "ohlcva");
        if (horizon < 1 || horizon > config.OutputPatchLen)
            throw new ArgumentOutOfRangeException(nameof(horizon),
                $"{where}horizon must lie in 1..{config.OutputPatchLen}");
    }

    private void ValidateDestination(int length, int entries, int horizon, string name)
    {
        if (length != BatchOutputLength(entries, horizon))
            throw new ArgumentException(
                $"{name} must hold exactly {BatchOutputLength(entries, horizon)} elements "
                + $"(entries x horizon x {config.NumQuantiles} quantiles); got {length}. Use {nameof(BatchOutputLength)}.",
                name);
    }
}

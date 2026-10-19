using System.Runtime.InteropServices;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.Timestral;

/// <summary>
/// Quantile forecasts from a Timestral checkpoint, in the shape shared with
/// <c>TimesFmForecaster</c> and <c>T0Forecaster</c>.
///
/// <para>Around the network it applies what the reference forecaster applies by default,
/// so published numbers are reproduced: each horizon patch is moved into the advancing
/// refine frame it was trained in, forecasts are capped at 10⁴ context scales beyond the
/// observed range, the forecast of the negated input is reflected and averaged in
/// (<see cref="SymmetricAveraging"/>), and series whose context is non-negative are
/// clipped at zero (<see cref="ClipNonNegative"/>). The mirrored input rides in the same
/// forward pass as the original, as further groups, rather than costing a second call.</para>
///
/// <para>Every series not marked known is a target, forecast jointly. Known-future series
/// are conditioned on, not forecast, and never read a target.</para>
/// </summary>
/// <remarks>Inference is not thread safe.</remarks>
public sealed class TimestralForecaster(TimestralModel model, Device device, string name = "")
    : IQuantileForecaster
{
    private const int CloseChannel = 3;    // open, high, low, close, volume, amount
    private const int Channels = 6;
    private const double QuantileCap = 1e4;

    private readonly TimestralConfig _c = model.Config;

    public TimestralConfig Config => _c;

    public TimestralModel Model => model;

    /// <summary>Checkpoint provenance, from the weights it was loaded from.</summary>
    public string CheckpointName => name;

    /// <inheritdoc/>
    public IReadOnlyList<double> Quantiles => _c.Quantiles;

    /// <summary>Longest horizon decoded, one block: 1024 steps for the published checkpoints.
    /// The reference continues block by block, feeding medians back; that is not ported.</summary>
    public int MaxHorizon => _c.MaxHorizon;

    /// <summary>Longest context, the trained length: 32768 for the published checkpoints.
    /// The reference silently keeps the most recent steps; a longer context is refused here.</summary>
    public int MaxContext => _c.ContextLength;

    /// <summary>Average with the reflected forecast of the negated input — every series,
    /// covariates included. On by default, as in the reference.</summary>
    public bool SymmetricAveraging { get; set; } = true;

    /// <summary>Clip at zero the forecast of a series whose observed context is all
    /// non-negative. On by default, as in the reference.</summary>
    public bool ClipNonNegative { get; set; } = true;

    /// <summary>Rows — series, summed over entries and doubled under
    /// <see cref="SymmetricAveraging"/> — in one forward pass. Variate masks grow with its
    /// square and time masks with the square of the context, so long contexts want less.</summary>
    public int MaxRowsPerPass { get; set; } = 64;

    public static TimestralForecaster Load(IModelWeights weights, Device device)
        => new(TimestralModel.Load(weights, device), device, weights.Name);

    /// <summary>Load from a snapshot directory holding <c>config.json</c> and
    /// <c>model.safetensors</c>.</summary>
    public static TimestralForecaster Load(string checkpointDir, Device device)
        => Load(new DirectoryWeights(checkpointDir, howToFetch:
            "Fetch the weights with scripts/fetch-timestral-checkpoint.sh, or download config.json and "
            + "model.safetensors from https://huggingface.co/mistralai/Timestral-209M (or another size); "
            + "5M and 28M also ship in Tsfm.Forecasting.Timestral.Weights.*."), device);

    /// <summary>
    /// Joint forecast over several series, any of which may be known into the future.
    /// </summary>
    /// <param name="series">Variate-major <c>[variates x total]</c>: series <c>v</c> at step
    /// <c>t</c> is <c>series[v * total + t]</c>. Steps from <paramref name="contextBars"/>
    /// on are read only for known variates, and only the first <paramref name="horizon"/>.
    /// NaN is a missing observation.</param>
    /// <param name="variates">Series supplied.</param>
    /// <param name="contextBars">Observed steps, 1..<see cref="MaxContext"/>.</param>
    /// <param name="horizon">Steps ahead, 1..<see cref="MaxHorizon"/>.</param>
    /// <param name="knownFuture">Per variate, or empty for none. A variate marked known is
    /// NOT a forecast target.</param>
    /// <param name="targetVariate">Which series the returned quantiles describe.</param>
    /// <returns><c>[horizon, 21]</c> in the units of the target series.</returns>
    public double[,] ForecastJoint(
        ReadOnlySpan<float> series, int variates, int contextBars, int horizon,
        ReadOnlySpan<bool> knownFuture = default, int targetVariate = 0)
    {
        ValidateHorizon(horizon);
        var total = ValidateJoint(series.Length, variates, contextBars, horizon, knownFuture, targetVariate, "");
        var result = new double[horizon, _c.NumQuantiles];
        var layout = new Layout(_c.PatchSize, contextBars, horizon, variates, Mirrored);
        var entry = AddJoint(layout, series, total, variates, contextBars, knownFuture, targetVariate);
        Run(layout, [entry], [0], MemoryMarshal.CreateSpan(ref result[0, 0], result.Length));
        return result;
    }

    /// <summary>Forward-return quantiles from OHLCVA bars, every channel a target.</summary>
    /// <param name="ohlcva">Row-major <c>[L x 6]</c>: open, high, low, close, volume, amount.</param>
    /// <param name="horizon">Steps ahead, 1..<see cref="MaxHorizon"/>.</param>
    /// <returns><c>[horizon, 21]</c> of cumulative return relative to the anchor close.</returns>
    public double[,] Forecast(ReadOnlySpan<float> ohlcva, int horizon)
    {
        ValidateHorizon(horizon);
        ValidateBars(ohlcva.Length, "");
        var result = new double[horizon, _c.NumQuantiles];
        var flat = MemoryMarshal.CreateSpan(ref result[0, 0], result.Length);
        var layout = new Layout(_c.PatchSize, ohlcva.Length / Channels, horizon, Channels, Mirrored);
        Run(layout, [AddBars(layout, ohlcva)], [0], flat);
        ToReturns(flat, ohlcva);
        return result;
    }

    /// <inheritdoc cref="IQuantileForecaster.BatchOutputLength"/>
    public int BatchOutputLength(int entries, int horizon) => entries * horizon * _c.NumQuantiles;

    /// <summary>
    /// <see cref="ForecastJoint"/> over many entries, each its own group, in passes of at
    /// most <see cref="MaxRowsPerPass"/> rows. Contexts may differ: shorter ones are
    /// left-padded with unobserved patches, which are neither keys nor positions, so an
    /// entry's forecast does not depend on its companions.
    /// </summary>
    /// <param name="batch">Entries, each with the constraints of <see cref="ForecastJoint"/>.</param>
    /// <param name="horizon">Steps ahead, shared.</param>
    /// <param name="quantiles">Receives <c>[entry][step][quantile]</c>; sized by
    /// <see cref="BatchOutputLength"/>.</param>
    public void ForecastJointBatch(ReadOnlySpan<JointSeries> batch, int horizon, Span<double> quantiles)
    {
        ValidateHorizon(horizon);
        ValidateDestination(quantiles.Length, batch.Length, horizon, nameof(quantiles));
        var totals = new int[batch.Length];
        var contexts = new int[batch.Length];
        var rows = new int[batch.Length];
        var variates = new int[batch.Length];
        for (var i = 0; i < batch.Length; i++)
        {
            var e = batch[i];
            totals[i] = ValidateJoint(e.Series.Length, e.Variates, e.ContextBars, horizon, e.KnownFuture.Span,
                e.TargetVariate, $"batch[{i}]: ");
            (contexts[i], variates[i], rows[i]) = (e.ContextBars, e.Variates, e.Variates * (Mirrored ? 2 : 1));
        }

        foreach (var pass in Passes(contexts, rows))
        {
            var layout = new Layout(_c.PatchSize, pass.Max(i => contexts[i]), horizon, pass.Sum(i => variates[i]),
                Mirrored);
            var entries = new Entry[pass.Length];
            for (var j = 0; j < pass.Length; j++)
            {
                var e = batch[pass[j]];
                entries[j] = AddJoint(layout, e.Series.Span, totals[pass[j]], e.Variates, e.ContextBars,
                    e.KnownFuture.Span, e.TargetVariate);
            }
            Run(layout, entries, pass, quantiles);
        }
    }

    /// <summary><see cref="Forecast"/> over many OHLCVA windows, batched as
    /// <see cref="ForecastJointBatch"/> is.</summary>
    /// <param name="ohlcva">One row-major <c>[L x 6]</c> window per entry.</param>
    /// <param name="horizon">Steps ahead, shared.</param>
    /// <param name="returns">Receives <c>[entry][step][quantile]</c> of cumulative return;
    /// sized by <see cref="BatchOutputLength"/>.</param>
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

        var rows = Enumerable.Repeat(Channels * (Mirrored ? 2 : 1), ohlcva.Length).ToArray();
        foreach (var pass in Passes(contexts, rows))
        {
            var layout = new Layout(_c.PatchSize, pass.Max(i => contexts[i]), horizon, pass.Length * Channels, Mirrored);
            var entries = new Entry[pass.Length];
            for (var j = 0; j < pass.Length; j++) entries[j] = AddBars(layout, ohlcva[pass[j]].Span);
            Run(layout, entries, pass, returns);
        }

        var block = horizon * _c.NumQuantiles;
        for (var i = 0; i < ohlcva.Length; i++) ToReturns(returns.Slice(i * block, block), ohlcva[i].Span);
    }

    private bool Mirrored => SymmetricAveraging;

    /// <summary>Entry indices ordered by context, cut wherever the next would exceed the row
    /// budget; an entry wider than the budget runs alone.</summary>
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

    /// <summary>An entry's target row in the layout.</summary>
    private readonly record struct Entry(int TargetRow);

    /// <summary>Targets first, then known-future rows: the reference's row order.</summary>
    private static Entry AddJoint(Layout layout, ReadOnlySpan<float> series, int total, int variates, int contextBars,
        ReadOnlySpan<bool> knownFuture, int targetVariate)
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
        return new Entry(target);
    }

    private static Entry AddBars(Layout layout, ReadOnlySpan<float> ohlcva)
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
        return new Entry(close);
    }

    private static void ToReturns(Span<double> prices, ReadOnlySpan<float> ohlcva)
    {
        var anchorClose = ohlcva[^(Channels - CloseChannel)];
        for (var i = 0; i < prices.Length; i++) prices[i] = prices[i] / anchorClose - 1.0;
    }

    /// <summary>
    /// One forward pass over every row of the layout — the originals and, under symmetric
    /// averaging, their negations — then the reference's post-processing per entry. Writes
    /// entry <c>j</c> at slot <c>slots[j]</c> of <paramref name="destination"/>.
    /// </summary>
    private void Run(Layout layout, Entry[] entries, ReadOnlySpan<int> slots, Span<double> destination)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var (p, q, h) = (_c.PatchSize, _c.NumQuantiles, layout.Horizon);
        var hp = layout.HorizonPatches;

        long[] dims = [layout.TotalRows, layout.Width];
        var values = tensor(layout.Values, dims).to(device);
        var observed = tensor(layout.Observed, dims).to(device);
        var groups = tensor(layout.Groups).to(device);
        var roles = tensor(layout.Roles).to(device);

        var raw = model.PredictBlock(values, observed, groups, roles, layout.ContextPatches, hp);

        // Only target rows are post-processed and read, so only their context is needed.
        var picked = entries.SelectMany(e => Mirrored ? new long[] { e.TargetRow, e.TargetRow + layout.Rows }
            : new long[] { e.TargetRow }).ToArray();
        var index = tensor(picked).to(device);
        var context = layout.ContextWidth;
        var ctx = values.index_select(0, index).narrow(1, 0, context);
        var ctxObs = observed.index_select(0, index).narrow(1, 0, context);
        var preds = Refine(raw.index_select(0, index), ctx, ctxObs, hp);
        preds = Cap(preds, ctx, ctxObs);
        if (preds.isnan().any().item<bool>())
            throw new InvalidOperationException("forecast produced NaN; the input is degenerate");

        // (picked, h·p, Q) -> host; the horizon rounds up to whole patches, so trim.
        var host = preds.narrow(1, 0, h).to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();
        var nonNegative = NonNegative(ctx, ctxObs).cpu().data<bool>().ToArray();
        var stride = h * q;
        for (var j = 0; j < entries.Length; j++)
        {
            var dst = destination.Slice(slots[j] * stride, stride);
            var a = (Mirrored ? 2 * j : j) * stride;
            var m = a + stride;
            for (var t = 0; t < h; t++)
            for (var k = 0; k < q; k++)
            {
                // The reference averages in float32: 0.5 * (q + flip(-mirrored)).
                var v = Mirrored ? 0.5f * (host[a + t * q + k] + -host[m + t * q + (q - 1 - k)]) : host[a + t * q + k];
                if (ClipNonNegative && nonNegative[Mirrored ? 2 * j : j]) v = MathF.Max(v, 0f);
                dst[t * q + k] = v;
            }
        }
    }

    /// <summary>Whether each row's observed context exists and is all non-negative.</summary>
    private static Tensor NonNegative(Tensor ctx, Tensor ctxObs)
        => ctxObs.any(-1) & (ctx.ge(0) | ~ctxObs).all(-1);

    /// <summary>
    /// Move a decoded block from the frame of the last context patch into the advancing
    /// refine frame it was trained in: horizon patch <c>j</c> is denormalized with the
    /// statistics of the observed context plus the block's medians for patches <c>0..j-1</c>.
    /// The first patch keeps the anchor frame. Welford states in float64, on the host.
    /// </summary>
    private Tensor Refine(Tensor quantiles, Tensor context, Tensor observed, int horizonPatches)
    {
        if (horizonPatches <= 1) return quantiles;
        using var scope = NewDisposeScope();
        var (p, rows) = (_c.PatchSize, (int)quantiles.shape[0]);
        var (loc, scale) = TimestralPreprocess.CausalPatchStats(context, observed, p, _c.MinimumScale);
        var loc0 = loc.select(1, loc.shape[1] - 1);
        var scale0 = scale.select(1, scale.shape[1] - 1);

        var median = quantiles.select(-1, _c.MedianIndex).to(ScalarType.Float64).cpu().data<double>().ToArray();
        var l0 = loc0.to(ScalarType.Float64).cpu().data<double>().ToArray();
        var s0 = scale0.to(ScalarType.Float64).cpu().data<double>().ToArray();
        var x = context.to(ScalarType.Float64).cpu().data<double>().ToArray();
        var w = observed.cpu().data<bool>().ToArray();

        var width = (int)context.shape[1];
        var locs = new float[rows * horizonPatches];
        var scales = new float[rows * horizonPatches];
        var clip = Math.Sqrt(float.MaxValue) / 1e3;
        for (var r = 0; r < rows; r++)
        {
            // The context's state, one patch at a time.
            double cnt = 0, mean = 0, m2 = 0;
            for (var start = 0; start < width; start += p)
            {
                double c = 0, s = 0;
                for (var t = start; t < start + p; t++) if (w[r * width + t]) { c++; s += x[r * width + t]; }
                var pm = c == 0 ? 0 : s / Math.Max(c, 1.0);
                double pm2 = 0;
                for (var t = start; t < start + p; t++)
                    if (w[r * width + t]) pm2 += (x[r * width + t] - pm) * (x[r * width + t] - pm);
                (cnt, mean, m2) = Combine(cnt, mean, m2, c, pm, pm2);
            }

            var offset = mean;   // medians are clamped relative to the context mean, so a large level survives
            for (var j = 0; j < horizonPatches; j++)
            {
                var sd = Math.Max(Math.Sqrt(Math.Max(m2 / Math.Max(cnt - 1.0, 1.0), 0.0)), _c.MinimumScale);
                locs[r * horizonPatches + j] = j == 0 ? (float)l0[r] : (float)mean;
                scales[r * horizonPatches + j] = j == 0 ? (float)s0[r] : (float)sd;

                double bs = 0;
                var folded = new double[p];
                for (var t = 0; t < p; t++)
                {
                    var sinhMedian = (median[(r * horizonPatches + j) * p + t] - l0[r]) / s0[r];
                    folded[t] = Math.Clamp(sinhMedian * sd + mean - offset, -clip, clip) + offset;
                    bs += folded[t];
                }
                var bm = bs / p;
                double bm2 = 0;
                foreach (var f in folded) bm2 += (f - bm) * (f - bm);
                (cnt, mean, m2) = Combine(cnt, mean, m2, p, bm, bm2);
            }
        }

        var lj = tensor(locs, [rows, horizonPatches]).to(quantiles.device).repeat_interleave(p, 1).unsqueeze(-1);
        var sj = tensor(scales, [rows, horizonPatches]).to(quantiles.device).repeat_interleave(p, 1).unsqueeze(-1);
        var norm = (quantiles - loc0.view(rows, 1, 1)) / scale0.view(rows, 1, 1);
        return TimestralPreprocess.ClampNonFinite(norm * sj + lj).MoveToOuterDisposeScope();
    }

    private static (double, double, double) Combine(double wa, double ma, double m2a, double wb, double mb, double m2b)
    {
        var w = wa + wb;
        if (w == 0) return (0, 0, 0);
        var delta = mb - ma;
        return (w, ma + delta * wb / w, m2a + m2b + delta * delta * wa * wb / w);
    }

    /// <summary>Clamp to <c>[min − k·s, max + k·s]</c> of the observed context, <c>s</c> the
    /// scale of its last patch and <c>k</c> 10⁴.</summary>
    private Tensor Cap(Tensor quantiles, Tensor context, Tensor observed)
    {
        using var scope = NewDisposeScope();
        var hi = nan_to_num(context.masked_fill(~observed, float.NegativeInfinity).amax([-1L], keepdim: true), 0, 0, 0);
        var lo = nan_to_num(context.masked_fill(~observed, float.PositiveInfinity).amin([-1L], keepdim: true), 0, 0, 0);
        var (_, scale) = TimestralPreprocess.CausalPatchStats(context, observed, _c.PatchSize, _c.MinimumScale);
        var band = scale.narrow(1, scale.shape[1] - 1, 1) * QuantileCap;
        return quantiles.clamp((lo - band).unsqueeze(-1), (hi + band).unsqueeze(-1)).MoveToOuterDisposeScope();
    }

    private void ValidateHorizon(int horizon)
    {
        if (horizon < 1 || horizon > MaxHorizon)
            throw new ArgumentOutOfRangeException(nameof(horizon), $"horizon must lie in 1..{MaxHorizon}");
    }

    private int ValidateJoint(int length, int variates, int contextBars, int horizon,
        ReadOnlySpan<bool> knownFuture, int targetVariate, string where)
    {
        if (variates < 1)
            throw new ArgumentOutOfRangeException(nameof(variates), $"{where}at least one series is required");
        if (length % variates != 0)
            throw new ArgumentException($"{where}series length {length} is not divisible by {variates} variates", "series");
        var total = length / variates;
        if (contextBars < 1 || contextBars > total)
            throw new ArgumentOutOfRangeException(nameof(contextBars),
                $"{where}contextBars must lie in 1..{total}; got {contextBars}");
        if (contextBars > MaxContext)
            throw new ArgumentOutOfRangeException(nameof(contextBars),
                $"{where}{CheckpointName} attends over at most {MaxContext} steps; got {contextBars}. Pass the most recent {MaxContext}.");
        if (targetVariate < 0 || targetVariate >= variates)
            throw new ArgumentOutOfRangeException(nameof(targetVariate), $"{where}no variate {targetVariate}");
        if (!knownFuture.IsEmpty && knownFuture.Length != variates)
            throw new ArgumentException($"{where}knownFuture must hold {variates} flags or be empty", nameof(knownFuture));
        if (!knownFuture.IsEmpty && knownFuture[targetVariate])
            throw new ArgumentException($"{where}the target variate cannot be known into the future", nameof(knownFuture));
        if (!knownFuture.IsEmpty && knownFuture.Contains(true))
        {
            if (!_c.CovariateEnabled)
                throw new ArgumentException($"{where}{CheckpointName} was not trained with covariates", nameof(knownFuture));
            if (horizon > total - contextBars)
                throw new ArgumentOutOfRangeException(nameof(horizon),
                    $"{where}horizon {horizon} exceeds the {total - contextBars} known steps supplied after the context");
        }
        return total;
    }

    private void ValidateBars(int length, string where)
    {
        if (length == 0 || length % Channels != 0)
            throw new ArgumentException($"{where}ohlcva must hold whole bars of {Channels} channels", "ohlcva");
        if (length / Channels > MaxContext)
            throw new ArgumentOutOfRangeException("ohlcva", $"{where}at most {MaxContext} bars; got {length / Channels}");
    }

    private void ValidateDestination(int length, int entries, int horizon, string name)
    {
        if (length != BatchOutputLength(entries, horizon))
            throw new ArgumentException(
                $"{name} must hold exactly {BatchOutputLength(entries, horizon)} elements "
                + $"(entries x horizon x {_c.NumQuantiles} quantiles); got {length}. Use {nameof(BatchOutputLength)}.",
                name);
    }

    /// <summary>
    /// Packed rows for one pass: every row's context right-aligned on a patch boundary, the
    /// longest rounded up to whole patches and the rest left-padded unobserved. A horizon
    /// region follows only when some row is a known-future covariate, carrying its values.
    /// Under symmetric averaging the negated copy of every row follows the originals, each
    /// copied group its own series.
    /// </summary>
    private sealed class Layout
    {
        public readonly int Rows, TotalRows, ContextPatches, HorizonPatches, Horizon, ContextWidth;
        private readonly int _patch, _capacity;
        private readonly bool _mirrored;
        private readonly List<(long Group, float[] Context, float[]? Future)> _rows = [];
        private long _group;

        public Layout(int patch, int maxContext, int horizon, int rows, bool mirrored)
        {
            (_patch, Horizon, Rows, _capacity, _mirrored) = (patch, horizon, rows, rows, mirrored);
            ContextPatches = (maxContext + patch - 1) / patch;
            HorizonPatches = (horizon + patch - 1) / patch;
            ContextWidth = ContextPatches * patch;
            TotalRows = mirrored ? 2 * rows : rows;
        }

        public long NextGroup() => _group++;

        public int AddRow(long group, ReadOnlySpan<float> context, ReadOnlySpan<float> future)
        {
            _rows.Add((group, context.ToArray(), future.IsEmpty ? null : future.ToArray()));
            return _rows.Count - 1;
        }

        private bool AnyKnown => _rows.Any(r => r.Future is not null);

        /// <summary>Patches the encoder spans: the context, and the horizon too when any row
        /// carries known-future values.</summary>
        public int Width => (ContextPatches + (AnyKnown ? HorizonPatches : 0)) * _patch;

        public float[] Values => Build().Values;
        public bool[] Observed => Build().Observed;
        public long[] Groups => Build().Groups;
        public long[] Roles => Build().Roles;

        private (float[] Values, bool[] Observed, long[] Groups, long[] Roles)? _built;

        private (float[] Values, bool[] Observed, long[] Groups, long[] Roles) Build()
        {
            if (_built is { } b) return b;
            if (_rows.Count != _capacity) throw new InvalidOperationException("layout is incomplete");
            var width = Width;
            var values = new float[TotalRows * width];
            var observed = new bool[TotalRows * width];
            var groups = new long[TotalRows];
            var roles = new long[TotalRows];
            for (var copy = 0; copy < (_mirrored ? 2 : 1); copy++)
            for (var i = 0; i < Rows; i++)
            {
                var (group, context, future) = _rows[i];
                var row = copy * Rows + i;
                var sign = copy == 0 ? 1f : -1f;
                groups[row] = group + copy * _group;
                roles[row] = future is null ? Role.Target : Role.KnownCovariate;
                var o = row * width;
                var pad = ContextWidth - context.Length;
                for (var t = 0; t < context.Length; t++) Write(o + pad + t, context[t] * sign);
                if (future is null) continue;
                for (var t = 0; t < future.Length; t++) Write(o + ContextWidth + t, future[t] * sign);
            }
            _built = (values, observed, groups, roles);
            return _built.Value;

            void Write(int at, float v)
            {
                if (!float.IsFinite(v)) return;    // NaN is missing; the reference reads isfinite
                values[at] = v;
                observed[at] = true;
            }
        }
    }
}

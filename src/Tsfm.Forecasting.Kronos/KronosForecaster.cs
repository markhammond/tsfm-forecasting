using System.Buffers;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

using Tsfm.Forecasting;

namespace Tsfm.Forecasting.Kronos;

/// <summary>
/// End-to-end inference: raw bars in, per-window summaries out.
///
/// <para>Tensors do not cross this boundary — callers pass spans and receive spans.</para>
///
/// <para><b>No key-value cache</b>, matching the reference: a cache changes the arithmetic
/// and forfeits checkable parity. Costs one full pass per decode step per rollout, so it
/// is affordable only at short horizons.</para>
/// </summary>
/// <remarks>Inference is not thread safe.</remarks>
public sealed class KronosForecaster : IDisposable
{
    private readonly KronosTokenizerEncoder _tokenizer;
    private readonly KronosModel _model;
    private readonly Device _device;

    /// <summary>
    /// Seeds each window's sampling stream from its anchor bar and series key. The default,
    /// <see cref="SamplingSeeds.AnchorTime"/>, ignores the key: every series anchored on a
    /// bar draws the same uniforms. See <see cref="SamplingSeeds"/> for the alternatives.
    /// Whatever the policy, a window's draws never depend on how windows were batched.
    /// </summary>
    public SamplingSeed Seed { get; set; } = SamplingSeeds.AnchorTime;

    /// <summary>Rows this slice produces, <c>L − K + 1</c>. Size buffers from this; a
    /// wrong count misaligns every row.</summary>
    /// <summary>
    /// Channels per bar, in order: open, high, low, close, volume, amount. Input is
    /// row-major, so bar <c>i</c> occupies <c>[i * Channels, i * Channels + Channels)</c>.
    ///
    /// <para>All of them are read and there is no mask, so a channel you lack must be
    /// filled rather than omitted — see <see cref="Infer"/> for the fills the reference
    /// uses. Amount is quote volume; where only base volume is known, the reference
    /// derives it as <c>volume * mean(open, high, low, close)</c>.</para>
    /// </summary>
    /// <example>
    /// <code>
    /// var ohlcva = new float[history.Count * KronosForecaster.Channels];
    /// var timeMs = new long[history.Count];
    /// for (var i = 0; i &lt; history.Count; i++)
    /// {
    ///     var c = history[i];
    ///     var o = i * KronosForecaster.Channels;
    ///     ohlcva[o + 0] = c.Open;
    ///     ohlcva[o + 1] = c.High;
    ///     ohlcva[o + 2] = c.Low;
    ///     ohlcva[o + 3] = c.Close;
    ///     ohlcva[o + 4] = c.Volume;
    ///     ohlcva[o + 5] = c.Volume * (c.Open + c.High + c.Low + c.Close) / 4f;
    ///     timeMs[i] = c.CloseTime.ToUnixTimeMilliseconds();
    /// }
    /// </code>
    /// </example>
    public const int Channels = 6;

    public static int OutputCount(int barCount, int contextBars) => barCount - contextBars + 1;

    private KronosForecaster(KronosTokenizerEncoder tokenizer, KronosModel model, Device device)
        => (_tokenizer, _model, _device) = (tokenizer, model, device);

    public string CheckpointName { get; private init; } = string.Empty;

    /// <summary>Longest context the loaded checkpoint can attend over, in bars.</summary>
    public int MaxContext { get; private init; }

    public static KronosForecaster Load(ICheckpoint checkpoint, Device device)
        => new(KronosTokenizerEncoder.FromCheckpoint(checkpoint, device),
               KronosModel.FromCheckpoint(checkpoint, device), device)
           { CheckpointName = checkpoint.Name, MaxContext = checkpoint.MaxContext };

    /// <summary>
    /// One inference over a contiguous slice, re-windowed internally so the payload is
    /// O(L) rather than O(L·K).
    /// </summary>
    /// <param name="ohlcva">
    /// Row-major <c>[L x 6]</c>: open, high, low, close, volume, amount.
    ///
    /// <para>All six are read. There is no mask, so a channel you do not have must be
    /// FILLED, and whatever you fill it with is read as data — it is not ignored. The
    /// reference implementation fills both volume and amount with <i>zero</i> when neither
    /// in available, and amount with <c>volume * mean(open, high, low, close)</c> —
    /// the mean of all four prices, not the close — when only volume is. NaN is rejected
    /// rather than treated as absent.</para>
    /// </param>
    /// <param name="barTimeMs">Close time per bar, Unix milliseconds, <c>[L]</c>. Cadence
    /// is taken from the spacing, so no interval need be passed.</param>
    /// <param name="lean">Receives <c>L − K + 1</c> medians of the projected mean return
    /// over the horizon, relative to the anchor close. Caller-allocated.</param>
    /// <param name="upCount">Receives the number of rollouts that finished positive.
    /// Divide by <paramref name="rollouts"/> for an empirical P(up).</param>
    /// <param name="dispersion">Receives the standard deviation across rollouts, or may be
    /// empty to skip computing it.</param>
    /// <param name="contextBars">Bars each window reads, K. Kronos-small tops out at 512.</param>
    /// <param name="horizon">Bars projected ahead. Costs a full forward pass each, per
    /// rollout — there is no key-value cache — so long horizons are expensive.</param>
    /// <param name="rollouts">Monte Carlo samples per window.</param>
    /// <param name="greedy">Take the argmax instead of sampling. Collapses dispersion.</param>
    /// <param name="temperature">Logit temperature before sampling.</param>
    /// <param name="topP">Nucleus cutoff.</param>
    /// <param name="batch">Windows per forward pass.</param>
    /// <param name="seriesKey">Identifies the series to <see cref="Seed"/>. The default policy
    /// ignores it; <see cref="SamplingSeeds.PerSeries"/> needs a distinct, stable key per
    /// instrument — a symbol hash, say.</param>
    /// <returns><c>L − K + 1</c> rows, the last anchored on the final bar. Forward
    /// timestamps are extrapolated from the bar interval, so no trailing bars are consumed.</returns>
    /// <remarks>Not thread safe.</remarks>
    public bool Infer(ReadOnlySpan<float> ohlcva, ReadOnlySpan<long> barTimeMs,
        Span<float> lean, Span<int> upCount, Span<float> dispersion,
        int contextBars, int horizon, int rollouts, bool greedy,
        float temperature, float topP, int batch = 8, long seriesKey = 0)
        => InferCore(ohlcva, barTimeMs, lean, upCount, dispersion, default, null,
            contextBars, horizon, rollouts, greedy, temperature, topP, batch, seriesKey);

    // CS1573 does not expand <inheritdoc>, though the IDE does, so the shared parameters
    // would have to be duplicated from Infer to satisfy a check the tooltips already pass.
#pragma warning disable CS1573
    /// <inheritdoc cref="Infer"/>
    /// <summary>
    /// As <see cref="Infer"/>, and additionally writes the projected OHLCVA path.
    /// </summary>
    /// <param name="path">Receives <c>rows x horizon x 6</c>, row-major, in the same channel
    /// order as the input. Sized by <see cref="PathLength"/>.</param>
    /// <param name="aggregate">Reduces the rollouts for one channel at one step to a single
    /// value; null uses <see cref="RolloutAggregators.Mean"/>, matching the reference, which
    /// averages across samples per element. Samples are ordered by rollout index, so
    /// <see cref="RolloutAggregators.Rollout"/> selects one whole path.</param>
    /// <remarks>
    /// The result NEED NOT satisfy <c>high >= max(open, close)</c> or
    /// <c>low &lt;= min(open, close)</c>. This is the model, not the aggregation: channels
    /// are decoded independently and neither this port nor the reference constrains them.
    /// A single rollout broke the ordering in roughly 30% to 70% of candles depending on
    /// the checkpoint, the smaller ones being worse; averaging across rollouts reduces that
    /// severalfold, because it cancels the per-channel noise rather than preventing it.
    /// Clamp after the fact if you need well-formed candles.
    ///
    /// <para>Measured on 5-minute BTC over one week (n=2016), the projected direction was
    /// not distinguishable from noise, and rollout spread mostly restated trailing realised
    /// volatility. Treat the path as a plausible sample, not a forecast, unless you have
    /// measured otherwise on your own instrument.</para>
    ///
    /// <para>Not thread safe.</para>
    /// </remarks>
    public bool InferPath(ReadOnlySpan<float> ohlcva, ReadOnlySpan<long> barTimeMs,
        Span<float> lean, Span<int> upCount, Span<float> dispersion, Span<float> path,
        int contextBars, int horizon, int rollouts, bool greedy,
        float temperature, float topP, RolloutAggregator? aggregate = null, int batch = 8, long seriesKey = 0)
        => InferCore(ohlcva, barTimeMs, lean, upCount, dispersion, path,
            aggregate ?? RolloutAggregators.Mean,
            contextBars, horizon, rollouts, greedy, temperature, topP, batch, seriesKey);
#pragma warning restore CS1573

    /// <summary>Elements <see cref="InferPath"/> writes: <c>(L - K + 1) * horizon * Channels</c>.</summary>
    public static int PathLength(int barCount, int contextBars, int horizon)
        => OutputCount(barCount, contextBars) * horizon * Channels;

#pragma warning disable CS1573
    /// <inheritdoc cref="InferPath"/>
    /// <summary>
    /// One forecast per series, anchored on each series' final bar — instruments at the
    /// latest bar, say — batched across series rather than along one. Row <c>i</c> equals
    /// the last row <see cref="InferPath"/> would produce for <c>series[i]</c> alone.
    ///
    /// <para>Rollouts and series share the forward pass: <paramref name="batch"/> series of
    /// <paramref name="rollouts"/> each make <c>batch * rollouts</c> rows. Each output reduces
    /// only its own series' rollouts.</para>
    ///
    /// <para>Sampling is seeded by <see cref="Seed"/>. Under the default, the anchor bar's
    /// timestamp alone, series anchored on the same bar draw the same uniforms — common
    /// random numbers: comparisons across instruments are less noisy, but their sampling
    /// errors are correlated. <see cref="SamplingSeeds.PerSeries"/> with distinct
    /// <see cref="KronosSeries.SeriesKey"/>s makes them independent.</para>
    /// </summary>
    /// <param name="series">Each with at least <paramref name="contextBars"/> bars.</param>
    /// <param name="lean">One row per series.</param>
    /// <param name="upCount">One row per series.</param>
    /// <param name="dispersion">One row per series, or empty.</param>
    /// <param name="path">Receives <c>series x horizon x 6</c>, or may be empty.</param>
    /// <param name="aggregate">Reduces each channel's rollouts at each step; null is the
    /// mean. Only applied when <paramref name="path"/> is requested.</param>
    /// <param name="batch">Series per forward pass.</param>
    public void InferBatch(ReadOnlySpan<KronosSeries> series,
        Span<float> lean, Span<int> upCount, Span<float> dispersion, Span<float> path,
        int contextBars, int horizon, int rollouts, bool greedy,
        float temperature, float topP, RolloutAggregator? aggregate = null, int batch = 8)
    {
        for (var i = 0; i < series.Length; i++)
        {
            var (o, t) = (series[i].Ohlcva.Length, series[i].BarTimeMs.Length);
            if (o != t * Channels)
                throw new ArgumentException($"series[{i}]: ohlcva must be L x 6 for its {t} bars", nameof(series));
            if (t < contextBars)
                throw new ArgumentException(
                    $"series[{i}]: {t} bars cannot fill a context of {contextBars}", nameof(series));
        }
        Drive(new Windows(series, contextBars), series.Length, lean, upCount, dispersion, path,
            path.IsEmpty ? null : aggregate ?? RolloutAggregators.Mean,
            contextBars, horizon, rollouts, greedy, temperature, topP, batch);
    }
#pragma warning restore CS1573

    private bool InferCore(ReadOnlySpan<float> ohlcva, ReadOnlySpan<long> barTimeMs,
        Span<float> lean, Span<int> upCount, Span<float> dispersion, Span<float> path,
        RolloutAggregator? aggregate,
        int contextBars, int horizon, int rollouts, bool greedy,
        float temperature, float topP, int batch, long seriesKey)
    {
        var length = barTimeMs.Length;
        if (ohlcva.Length != length * Channels)
            throw new ArgumentException("ohlcva must be L x 6", nameof(ohlcva));

        var count = OutputCount(length, contextBars);
        if (contextBars <= MaxContext && count <= 0) return false;
        Drive(new Windows(ohlcva, barTimeMs, contextBars, seriesKey), count, lean, upCount, dispersion, path, aggregate,
            contextBars, horizon, rollouts, greedy, temperature, topP, batch);
        return true;
    }

    /// <summary>Evaluate <paramref name="count"/> windows, <paramref name="batch"/> per
    /// forward pass, writing output row <c>i</c> for window <c>i</c>.</summary>
    private void Drive(in Windows source, int count,
        Span<float> lean, Span<int> upCount, Span<float> dispersion, Span<float> path,
        RolloutAggregator? aggregate,
        int contextBars, int horizon, int rollouts, bool greedy,
        float temperature, float topP, int batch)
    {
        var k = contextBars;
        var h = horizon;

        // Upstream truncates silently, which reads as a working call that quietly ignored
        // the bars beyond the limit. Refuse instead, and say what the limit is.
        if (k > MaxContext)
            throw new ArgumentOutOfRangeException(nameof(contextBars),
                $"{CheckpointName} attends over at most {MaxContext} bars; got {k}.");

        if (lean.Length != count || upCount.Length != count)
            throw new ArgumentException(
                $"lean and upCount must each hold exactly {count} rows; " +
                $"got {lean.Length}/{upCount.Length}. Use {nameof(OutputCount)}.");

        // Dispersion is diagnostic. An empty span skips the standard deviation entirely.
        if (!dispersion.IsEmpty && dispersion.Length != count)
            throw new ArgumentException(
                $"dispersion must hold exactly {count} rows or be empty; got {dispersion.Length}.");

        if (!path.IsEmpty && path.Length != count * h * Channels)
            throw new ArgumentException(
                $"path must hold exactly {count * h * Channels} elements (rows x horizon x 6); " +
                $"got {path.Length}. Use {nameof(PathLength)}.");

        lean.Fill(float.NaN);
        if (!dispersion.IsEmpty) dispersion.Fill(float.NaN);
        if (!path.IsEmpty) path.Fill(float.NaN);
        upCount.Clear();
        if (count == 0) return;

        // Size scratch for a full batch and reuse it across the slice; per-batch
        // allocation dominates a materialisation-sized run.
        var maxRows = batch * rollouts;
        var x = ArrayPool<float>.Shared.Rent(maxRows * k * Channels);
        var stamp = ArrayPool<float>.Shared.Rent(maxRows * (k + h) * 5);
        var mean = ArrayPool<float>.Shared.Rent(batch * Channels);
        var scale = ArrayPool<float>.Shared.Rent(batch * Channels);
        var anchor = ArrayPool<float>.Shared.Rent(batch);
        var uniforms = ArrayPool<float>.Shared.Rent(batch * rollouts * h * 2);
        var pick = ArrayPool<float>.Shared.Rent(maxRows);
        var samples = ArrayPool<float>.Shared.Rent(rollouts);
        var per = ArrayPool<double>.Shared.Rent(rollouts);

        try
        {
            using var _ = no_grad();
            for (var start = 0; start < count; start += batch)
            {
                var take = Math.Min(batch, count - start);
                EvaluateBatch(source, start, take,
                    k, h, rollouts, greedy, temperature, topP,
                    lean, upCount, dispersion, path, aggregate,
                    samples.AsSpan(0, rollouts), per.AsSpan(0, rollouts),
                    x, stamp, mean, scale, anchor, uniforms, pick);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(stamp);
            ArrayPool<float>.Shared.Return(mean);
            ArrayPool<float>.Shared.Return(scale);
            ArrayPool<float>.Shared.Return(anchor);
            ArrayPool<float>.Shared.Return(uniforms);
            ArrayPool<float>.Shared.Return(pick);
            ArrayPool<float>.Shared.Return(samples);
            ArrayPool<double>.Shared.Return(per);
        }
    }

    /// <summary>
    /// Where window <c>i</c> lives: either the <c>i</c>-th sliding window along one series,
    /// or the final window of the <c>i</c>-th of many. Either way it reads K bars from
    /// <see cref="Origin"/> of <see cref="Ohlcva"/>, anchored on the last of them.
    /// </summary>
    private readonly ref struct Windows
    {
        private readonly ReadOnlySpan<float> _ohlcva;
        private readonly ReadOnlySpan<long> _times;
        private readonly ReadOnlySpan<KronosSeries> _series;
        private readonly int _k;
        private readonly bool _many;
        private readonly long _key;

        public Windows(ReadOnlySpan<float> ohlcva, ReadOnlySpan<long> times, int k, long key)
        {
            _ohlcva = ohlcva;
            _times = times;
            _k = k;
            _key = key;
        }

        public Windows(ReadOnlySpan<KronosSeries> series, int k)
        {
            _series = series;
            _k = k;
            _many = true;
        }

        public ReadOnlySpan<float> Ohlcva(int i) => _many ? _series[i].Ohlcva.Span : _ohlcva;

        public ReadOnlySpan<long> Times(int i) => _many ? _series[i].BarTimeMs.Span : _times;

        public int Origin(int i) => _many ? _series[i].BarTimeMs.Length - _k : i;

        /// <summary>Bar interval, from the series' first spacing.</summary>
        public long SeriesKey(int i) => _many ? _series[i].SeriesKey : _key;

        public long StepMs(int i)
        {
            var t = Times(i);
            return t.Length > 1 ? t[1] - t[0] : 0L;
        }
    }

    private void EvaluateBatch(
        in Windows source, int start, int take, int k, int h, int rollouts, bool greedy, float temperature, float topP,
        Span<float> lean, Span<int> upCount, Span<float> dispersion,
        Span<float> path, RolloutAggregator? aggregate, Span<float> samples, Span<double> per,
        float[] x, float[] stamp, float[] mean, float[] scale, float[] anchor,
        float[] uniforms, float[] pick)
    {
        using var scope = NewDisposeScope();
        var bn = take * rollouts;

        // Normalisation is window-local and causal: statistics come from the same K bars
        // the model reads.
        for (var w = 0; w < take; w++)
        {
            var ohlcva = source.Ohlcva(start + w);
            var times = source.Times(start + w);
            var origin = source.Origin(start + w);
            var stepMs = source.StepMs(start + w);
            NormaliseWindow(ohlcva, origin, k, mean.AsSpan(w * Channels, Channels), scale.AsSpan(w * Channels, Channels));
            anchor[w] = ohlcva[(origin + k - 1) * Channels + 3];               // close of the anchor bar

            // Calendar stamps for the window and its extrapolated horizon, written once and
            // shared by every rollout.
            var first = stamp.AsSpan(w * rollouts * (k + h) * 5, (k + h) * 5);
            for (var t = 0; t < k; t++) WriteStamp(first, t, times[origin + t]);
            for (var j = 0; j < h; j++)                                  // forward stamps: extrapolated
                WriteStamp(first, k + j, times[origin + k - 1] + stepMs * (j + 1));

            for (var r = 0; r < rollouts; r++)
            {
                var row = w * rollouts + r;
                for (var t = 0; t < k; t++)
                for (var c = 0; c < Channels; c++)
                {
                    var v = (ohlcva[(origin + t) * Channels + c] - mean[w * Channels + c]) / (scale[w * Channels + c] + 1e-5f);
                    x[(row * k + t) * Channels + c] = Math.Clamp(v, -5f, 5f);
                }
                if (r > 0) first.CopyTo(stamp.AsSpan(row * (k + h) * 5, (k + h) * 5));
            }
        }

        // Slice explicitly. ArrayPool returns arrays longer than requested and the final
        // batch fills fewer rows; reshaping the backing array reinterprets stale values.
        var xt = tensor(new Memory<float>(x, 0, bn * k * Channels), [bn, (long)k, (long)Channels]).to(_device);
        var st = tensor(new Memory<float>(stamp, 0, bn * (k + h) * 5), [bn, k + h, 5L]).to(_device);

        var (s1, s2) = _tokenizer.forward(xt);
        var pre = cat([s1, zeros(bn, h, ScalarType.Int64, _device)], dim: 1);
        var post = cat([s2, zeros(bn, h, ScalarType.Int64, _device)], dim: 1);

        // Draw before the loop: sampling must be a pure function of the bar, not of how
        // windows were batched.
        if (!greedy) DrawUniforms(source, Seed, start, take, k, rollouts, h, uniforms);
        for (var step = 0; step < h; step++)
        {
            var len = k + step;
            var (logits, context) = _model.DecodeS1(
                pre.narrow(1, 0, len), post.narrow(1, 0, len), st.narrow(1, 0, len).contiguous());
            var a = Pick(logits.select(1, len - 1), greedy, temperature, topP, uniforms, step * 2, bn, rollouts, h, pick);
            var l2 = _model.DecodeS2(context, a);
            var b = Pick(l2.select(1, len - 1), greedy, temperature, topP, uniforms, step * 2 + 1, bn, rollouts, h, pick);
            pre.select(1, len).copy_(a.squeeze(-1));
            post.select(1, len).copy_(b.squeeze(-1));
        }

        // Decode the whole sequence and keep the generated tail. All six channels are
        // materialised only when a path is requested; otherwise just the close.
        var decoded = _tokenizer.Decode(pre, post).narrow(1, k, h);
        var closes = decoded.select(-1, 3).to(ScalarType.Float32).cpu().data<float>().ToArray();
        var all = path.IsEmpty
            ? []
            : decoded.reshape(bn * h * Channels).to(ScalarType.Float32).cpu().data<float>().ToArray();

        for (var w = 0; w < take; w++)
        {
            var mu = mean[w * Channels + 3];
            var sd = scale[w * Channels + 3];
            var c0 = anchor[w];
            for (var r = 0; r < rollouts; r++)
            {
                double sum = 0;
                for (var j = 0; j < h; j++)
                    sum += closes[(w * rollouts + r) * h + j] * (sd + 1e-5f) + mu;
                per[r] = sum / h / c0 - 1.0;
            }
            per.Sort();
            var up = 0;
            foreach (var v in per) if (v > 0) up++;
            // Median, not mean: the mean converges to the greedy projection and outlier
            // rollouts drag it.
            lean[start + w] = (float)(rollouts % 2 == 1
                ? per[rollouts / 2]
                : 0.5 * (per[rollouts / 2 - 1] + per[rollouts / 2]));
            upCount[start + w] = up;
            if (!dispersion.IsEmpty) dispersion[start + w] = rollouts > 1 ? (float)StdDev(per) : 0f;

            if (path.IsEmpty) continue;
            var reduce = aggregate!;
            var window = path.Slice((start + w) * h * Channels, h * Channels);
            for (var j = 0; j < h; j++)
            for (var ch = 0; ch < Channels; ch++)
            {
                // Ordered by rollout index, so a delegate returning samples[i] picks one
                // whole path and keeps the candle internally consistent.
                for (var r = 0; r < rollouts; r++)
                    samples[r] = all[((w * rollouts + r) * h + j) * Channels + ch]
                                 * (scale[w * Channels + ch] + 1e-5f) + mean[w * Channels + ch];
                window[j * Channels + ch] = reduce(samples);
            }
        }
    }

    private static double StdDev(ReadOnlySpan<double> v)
    {
        double m = 0;
        foreach (var t in v) m += t;
        m /= v.Length;
        double acc = 0;
        foreach (var t in v) acc += (t - m) * (t - m);
        return Math.Sqrt(acc / (v.Length - 1));
    }

    /// <summary>Per-channel mean and population standard deviation, ddof = 0 as
    /// <c>np.std</c>.</summary>
    private static void NormaliseWindow(
        ReadOnlySpan<float> ohlcva, int origin, int k, Span<float> mean, Span<float> sd)
    {
        for (var c = 0; c < Channels; c++)
        {
            double sum = 0;
            for (var t = 0; t < k; t++) sum += ohlcva[(origin + t) * Channels + c];
            var m = sum / k;
            double acc = 0;
            for (var t = 0; t < k; t++)
            {
                var d = ohlcva[(origin + t) * Channels + c] - m;
                acc += d * d;
            }
            mean[c] = (float)m;
            sd[c] = (float)Math.Sqrt(acc / k);
        }
    }

    /// <summary>One uniform per (window, rollout, decode step, head), seeded from the
    /// window's anchor bar. SplitMix64, so the stream cannot shift with a framework
    /// version.</summary>
    private static void DrawUniforms(
        in Windows source, SamplingSeed seed, int start, int take, int k, int rollouts, int h, float[] u)
    {
        var perWindow = rollouts * h * 2;
        for (var w = 0; w < take; w++)
        {
            var anchorMs = source.Times(start + w)[source.Origin(start + w) + k - 1];
            var state = (ulong)seed(anchorMs, source.SeriesKey(start + w));
            for (var i = 0; i < perWindow; i++)
            {
                state += 0x9E3779B97F4A7C15UL;
                var z = state;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                z ^= z >> 31;
                u[w * perWindow + i] = (float)((z >> 11) * (1.0 / 9007199254740992.0));
            }
        }
    }

    /// <summary>Select a token. Greedy takes the argmax; sampling uses inverse CDF against
    /// a pre-drawn uniform, making the draw independent of batch grouping. Distributionally
    /// identical to the reference; the RNG stream deliberately is not.</summary>
    private static Tensor Pick(Tensor logits, bool greedy, float temperature, float topP,
        float[] uniforms, int slot, int bn, int rollouts, int h, float[] pick)
    {
        using var scope = NewDisposeScope();
        if (greedy) return logits.argmax(-1, keepdim: true).MoveToOuterDisposeScope();

        var scaled = logits / Math.Max(temperature, 1e-6f);
        var probs = F.softmax(scaled, -1);

        if (topP < 1f)
        {
            var (sorted, indices) = probs.sort(dim: -1, descending: true);
            var cum = sorted.cumsum(-1);
            var keep = (cum - sorted).le(topP);
            sorted = sorted * keep.to(sorted.dtype);
            probs = zeros_like(probs).scatter_(-1, indices, sorted);
        }

        probs = probs / probs.sum(-1, keepdim: true);
        var cdf = probs.cumsum(-1);

        // Window-major layout: the row index divides out to its window.
        var perWindow = rollouts * h * 2;
        for (var i = 0; i < bn; i++) pick[i] = uniforms[(i / rollouts) * perWindow + (i % rollouts) * h * 2 + slot];
        var u = from_array(pick.AsSpan(0, bn).ToArray(), ScalarType.Float32)
            .reshape(bn, 1).to(logits.device);

        var idx = searchsorted(cdf, u);
        return idx.clamp(0, cdf.shape[^1] - 1).MoveToOuterDisposeScope();
    }

    /// <summary>Calendar features: minute, hour, weekday, day-of-month, month.
    /// <b>Weekday is Monday-zero</b>, as pandas. <see cref="DayOfWeek"/> is Sunday-zero;
    /// using it directly shifts every weekday embedding by one, silently.</summary>
    public static void WriteStamp(Span<float> dst, int row, long unixMs)
    {
        var t = DateTimeOffset.FromUnixTimeMilliseconds(unixMs).UtcDateTime;
        var o = row * 5;
        dst[o + 0] = t.Minute;
        dst[o + 1] = t.Hour;
        dst[o + 2] = ((int)t.DayOfWeek + 6) % 7;   // Sunday-zero -> Monday-zero
        dst[o + 3] = t.Day;
        dst[o + 4] = t.Month;
    }

    /// <summary>
    ///     FNV-1a over the bar's timestamp. Never carried between bars, so a sliding
    ///     pass and a one-shot pass agree.</summary>
    /// <remarks>
    ///     Deterministically seeded from the timestamp
    /// </remarks>
    public static long SeedFor(long closeTimeMs)
    {
        var hash = 0xcbf29ce484222325UL;
        for (var i = 0; i < 8; i++)
        {
            hash ^= (byte)(closeTimeMs >> (i * 8));
            hash *= 0x100000001b3UL;
        }
        return (long)(hash & 0x7FFFFFFFFFFFFFFFUL);
    }

    public void Dispose()
    {
        _tokenizer.Dispose();
        _model.Dispose();
    }
}

/// <summary>One series for <see cref="KronosForecaster.InferBatch"/>: row-major
/// <c>[L x 6]</c> bars and their close times. Memory rather than spans so series can sit in
/// an array; slicing a larger buffer copies nothing.</summary>
/// <param name="Ohlcva">Row-major <c>[L x 6]</c>: open, high, low, close, volume, amount.</param>
/// <param name="BarTimeMs">Close time per bar, Unix milliseconds, <c>[L]</c>.</param>
/// <param name="SeriesKey">Identifies the series to <see cref="KronosForecaster.Seed"/>;
/// ignored by the default policy. Keep it stable across calls — a symbol hash, not a batch
/// index — or a series' draws change with the company it is batched in.</param>
public readonly record struct KronosSeries(ReadOnlyMemory<float> Ohlcva, ReadOnlyMemory<long> BarTimeMs, long SeriesKey = 0);

/// <summary>A sampling seed for one window, from its anchor bar's close time and its
/// series key. Must be a pure function: draws are reproduced from it.</summary>
public delegate long SamplingSeed(long anchorCloseTimeMs, long seriesKey);

/// <summary>
/// Seeding policies for <see cref="KronosForecaster.Seed"/>.
///
/// <para>The choice is between common and independent random numbers across series. Common
/// ones (the default) make a cross-sectional comparison — ranking instruments by P(up), say
/// — less noisy, because every instrument's sampling error moves together; but for the same
/// reason an average over instruments does not shrink that error. Independent ones behave as
/// separate experiments.</para>
/// </summary>
public static class SamplingSeeds
{
    /// <summary>The anchor bar alone, ignoring the series: series anchored on one bar share
    /// draws. The behaviour of 0.1.x and the default.</summary>
    public static readonly SamplingSeed AnchorTime = (anchorMs, _) => KronosForecaster.SeedFor(anchorMs);

    /// <summary>The anchor bar mixed with the series key: independent streams for distinct
    /// keys, and the same stream for the same series and bar on every call.</summary>
    public static readonly SamplingSeed PerSeries = (anchorMs, key) => Mix(KronosForecaster.SeedFor(anchorMs), key);

    /// <summary><paramref name="inner"/> mixed with a run-level salt: a fresh, still
    /// reproducible, set of draws per salt — for independent replications of a whole run.</summary>
    public static SamplingSeed Salted(SamplingSeed inner, long salt) => (anchorMs, key) => Mix(inner(anchorMs, key), salt);

    /// <summary>SplitMix64's finaliser over the pair, kept non-negative like
    /// <see cref="KronosForecaster.SeedFor"/>.</summary>
    private static long Mix(long seed, long key)
    {
        var z = (ulong)seed ^ ((ulong)key * 0x9E3779B97F4A7C15UL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (long)(z & 0x7FFFFFFFFFFFFFFFUL);
    }
}

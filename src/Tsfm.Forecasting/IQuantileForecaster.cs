namespace Tsfm.Forecasting;

/// <summary>
/// One entry of a batched joint forecast: the arguments of
/// <see cref="IQuantileForecaster.ForecastJoint"/> other than the horizon, which a batch
/// shares. Memory rather than spans so entries can sit in an array; slicing one large
/// buffer into many entries copies nothing.
/// </summary>
/// <param name="Series">Variate-major <c>[Variates x total]</c>, as for
/// <see cref="IQuantileForecaster.ForecastJoint"/>.</param>
/// <param name="Variates">Series in this entry.</param>
/// <param name="ContextBars">Observed steps.</param>
/// <param name="KnownFuture">Per variate, or empty for none.</param>
/// <param name="TargetVariate">Which series the entry's quantiles describe.</param>
public readonly record struct JointSeries(
    ReadOnlyMemory<float> Series, int Variates, int ContextBars,
    ReadOnlyMemory<bool> KnownFuture = default, int TargetVariate = 0);

/// <summary>
/// A model that emits a quantile forecast for one series of a jointly attended set, any of
/// which may be known into the future. Implemented by the TimesFM and t0 adapters, so one
/// call site can drive either.
///
/// <para>Shapes are shared; constraints are not. Each implementation documents its own
/// limits on context, horizon and variate count — TimesFM, for instance, needs whole
/// patches where t0 pads.</para>
///
/// <para>The batch methods are the single calls over many independent entries, sharing a
/// horizon. Entries never attend to one another, so a batched result equals the single
/// call's up to float rounding; what a batch buys is fewer, larger forward passes. Results
/// land in a caller-supplied span, <c>[entry][step][quantile]</c> row-major.</para>
/// </summary>
/// <remarks>Implementations are not thread safe.</remarks>
public interface IQuantileForecaster
{
    /// <summary>Levels of the returned quantile columns, ascending — the checkpoint's own
    /// trained grid, so its length varies between checkpoints.</summary>
    IReadOnlyList<double> Quantiles { get; }

    /// <summary>Joint forecast over several series.</summary>
    /// <param name="series">Variate-major <c>[variates x total]</c>: series <c>v</c> at
    /// step <c>t</c> is <c>series[v * total + t]</c>, where <c>total = series.Length /
    /// variates</c>. The future region of a variate not marked known is never read.</param>
    /// <param name="variates">Series supplied.</param>
    /// <param name="contextBars">Observed steps.</param>
    /// <param name="horizon">Steps ahead to return.</param>
    /// <param name="knownFuture">Per variate: true where its future region holds real values
    /// the model may read. Empty means none. A variate marked known is not a target.</param>
    /// <param name="targetVariate">Which series the returned quantiles describe.</param>
    /// <returns><c>[horizon, Quantiles.Count]</c> in the units of the target series.</returns>
    double[,] ForecastJoint(
        ReadOnlySpan<float> series, int variates, int contextBars, int horizon,
        ReadOnlySpan<bool> knownFuture = default, int targetVariate = 0);

    /// <summary>Forward-return quantiles of the close from OHLCVA bars, every channel a
    /// target.</summary>
    /// <param name="ohlcva">Row-major <c>[L x 6]</c>: open, high, low, close, volume, amount.</param>
    /// <param name="horizon">Steps ahead to return.</param>
    /// <returns><c>[horizon, Quantiles.Count]</c> of cumulative return relative to the anchor close.</returns>
    double[,] Forecast(ReadOnlySpan<float> ohlcva, int horizon);

    /// <summary><see cref="ForecastJoint"/> over many entries.</summary>
    /// <param name="batch">Independent entries; each may differ in variates, context and
    /// known-future pattern, as far as the implementation allows.</param>
    /// <param name="horizon">Steps ahead, shared by every entry.</param>
    /// <param name="quantiles">Receives <c>[entry][step][quantile]</c>; sized by
    /// <see cref="BatchOutputLength"/>.</param>
    void ForecastJointBatch(ReadOnlySpan<JointSeries> batch, int horizon, Span<double> quantiles);

    /// <summary><see cref="Forecast"/> over many OHLCVA windows — instruments, or anchors
    /// along one history.</summary>
    /// <param name="ohlcva">One row-major <c>[L x 6]</c> window per entry.</param>
    /// <param name="horizon">Steps ahead, shared by every entry.</param>
    /// <param name="returns">Receives <c>[entry][step][quantile]</c> of cumulative return;
    /// sized by <see cref="BatchOutputLength"/>.</param>
    void ForecastBatch(ReadOnlySpan<ReadOnlyMemory<float>> ohlcva, int horizon, Span<double> returns);

    /// <summary>Elements a batch of <paramref name="entries"/> writes. A wrong size is
    /// refused rather than silently misaligning every entry.</summary>
    int BatchOutputLength(int entries, int horizon) => entries * horizon * Quantiles.Count;
}

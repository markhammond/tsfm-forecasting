namespace Tsfm.Forecasting;

/// <summary>
/// A model that emits a quantile forecast for one series of a jointly attended set, any of
/// which may be known into the future. Implemented by the TimesFM and t0 adapters, so one
/// call site can drive either.
///
/// <para>Shapes are shared; constraints are not. Each implementation documents its own
/// limits on context, horizon and variate count — TimesFM, for instance, needs whole
/// patches where t0 pads.</para>
/// </summary>
/// <remarks>Implementations are not thread safe.</remarks>
public interface IQuantileForecaster
{
    /// <summary>Levels of the returned quantile columns, ascending — the checkpoint's own
    /// trained grid, so its length varies between checkpoints.</summary>
    IReadOnlyList<double> Quantiles { get; }

    /// <summary>Joint forecast over several series.</summary>
    /// <param name="series">Variate-major <c>[variates x (contextBars + horizon)]</c>: series
    /// <c>v</c> at step <c>t</c> is <c>series[v * (contextBars + horizon) + t]</c>. The future
    /// region of a variate not marked known is never read.</param>
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
}

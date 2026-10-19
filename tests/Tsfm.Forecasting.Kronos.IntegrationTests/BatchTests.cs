using Tsfm.Forecasting.Kronos.Weights;

using Xunit;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.Kronos.IntegrationTests;

/// <summary>
/// <see cref="KronosForecaster.InferBatch"/> against the single-series path it batches.
/// Sampling is seeded by each window's anchor bar, so a sampled batch must reproduce the
/// single-series rows exactly, not merely in distribution.
/// </summary>
public class BatchTests : IDisposable
{
    private const int Context = 64, Horizon = 3, Rollouts = 5;

    private readonly KronosForecaster _forecaster =
        KronosForecaster.Load(KronosMini.Instance, new Device("cpu"));

    public void Dispose() => _forecaster.Dispose();

    [Fact]
    public void EachRowEqualsTheLastRowOfItsSeriesAlone()
    {
        // Three instruments of different lengths and levels, on different bar grids.
        KronosSeries[] series = [Synthetic(70, 100f, 0), Synthetic(64, 2500f, 7), Synthetic(90, 0.8f, 3)];
        var n = series.Length;
        var (lean, up, disp) = (new float[n], new int[n], new float[n]);
        var path = new float[n * Horizon * 6];

        _forecaster.InferBatch(series, lean, up, disp, path, Context, Horizon, Rollouts,
            greedy: false, 1f, 0.9f, batch: 2);

        for (var i = 0; i < n; i++)
        {
            var bars = series[i].BarTimeMs.Length;
            var rows = KronosForecaster.OutputCount(bars, Context);
            var (l, u, d) = (new float[rows], new int[rows], new float[rows]);
            var p = new float[KronosForecaster.PathLength(bars, Context, Horizon)];
            _forecaster.InferPath(series[i].Ohlcva.Span, series[i].BarTimeMs.Span, l, u, d, p,
                Context, Horizon, Rollouts, greedy: false, 1f, 0.9f);

            Assert.Equal(l[^1], lean[i]);
            Assert.Equal(u[^1], up[i]);
            Assert.Equal(d[^1], disp[i]);
            Assert.Equal(p.AsSpan(p.Length - Horizon * 6).ToArray(), path.AsSpan(i * Horizon * 6, Horizon * 6).ToArray());
        }
    }

    /// <summary>Two instruments with identical bars on the same grid: the default policy
    /// draws the same uniforms for both, PerSeries draws independently, and either way a
    /// batched row still equals its single-series call with the same key.</summary>
    [Fact]
    public void SeedPolicyDecidesWhetherSeriesShareDraws()
    {
        var a = Synthetic(70, 100f, 0) with { SeriesKey = 11 };
        var b = a with { SeriesKey = 22 };
        float[] Leans(SamplingSeed seed)
        {
            _forecaster.Seed = seed;
            var lean = new float[2];
            var path = new float[2 * Horizon * 6];
            _forecaster.InferBatch([a, b], lean, new int[2], default, path, Context, Horizon, Rollouts,
                greedy: false, 1f, 1f);
            return [path[3], path[Horizon * 6 + 3]];       // first step's close, each series
        }

        try
        {
            var common = Leans(SamplingSeeds.AnchorTime);
            Assert.Equal(common[0], common[1]);

            var independent = Leans(SamplingSeeds.PerSeries);
            Assert.NotEqual(independent[0], independent[1]);

            // Batch invariance holds under the policy: B alone, keyed 22, reproduces its row.
            var bars = b.BarTimeMs.Length;
            var rows = KronosForecaster.OutputCount(bars, Context);
            var p = new float[KronosForecaster.PathLength(bars, Context, Horizon)];
            _forecaster.InferPath(b.Ohlcva.Span, b.BarTimeMs.Span, new float[rows], new int[rows], default, p,
                Context, Horizon, Rollouts, greedy: false, 1f, 1f, seriesKey: 22);
            Assert.Equal(independent[1], p[p.Length - Horizon * 6 + 3]);

            var salted = Leans(SamplingSeeds.Salted(SamplingSeeds.AnchorTime, 1));
            Assert.Equal(salted[0], salted[1]);
            Assert.NotEqual(common[0], salted[0]);
        }
        finally
        {
            _forecaster.Seed = SamplingSeeds.AnchorTime;
        }
    }

    [Fact]
    public void TooShortASeriesIsRefused()
    {
        KronosSeries[] series = [Synthetic(Context - 1, 100f, 0)];
        var ex = Assert.Throws<ArgumentException>(() => _forecaster.InferBatch(series,
            new float[1], new int[1], default, default, Context, Horizon, Rollouts, greedy: true, 1f, 1f));
        Assert.Contains("series[0]", ex.Message);
    }

    private static KronosSeries Synthetic(int bars, float level, int offsetBars)
    {
        var ohlcva = new float[bars * 6];
        var times = new long[bars];
        var t0 = new DateTimeOffset(2026, 3, 2, 0, 0, 0, TimeSpan.Zero).ToUnixTimeMilliseconds();
        for (var i = 0; i < bars; i++)
        {
            var mid = level * (1f + 0.01f * MathF.Sin((i + offsetBars) / 9f));
            ohlcva[i * 6 + 0] = mid * 0.999f;
            ohlcva[i * 6 + 1] = mid * 1.002f;
            ohlcva[i * 6 + 2] = mid * 0.997f;
            ohlcva[i * 6 + 3] = mid * 1.001f;
            ohlcva[i * 6 + 4] = 50f + i % 5;
            ohlcva[i * 6 + 5] = ohlcva[i * 6 + 4] * mid;
            times[i] = t0 + (i + offsetBars) * 300_000L;
        }
        return new KronosSeries(ohlcva, times);
    }
}

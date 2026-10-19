using Xunit;

namespace Tsfm.Forecasting.Timestral.Tests;

/// <summary>Batched calls against the single calls they batch, on the tiny model. A pass
/// with a known-future entry widens every row's encoder over the horizon; entries without
/// one must be unaffected by that.</summary>
public class BatchTests
{
    /// <summary>Equal to float32 rounding: a few ulps of the value, or of 1 near zero.
    /// Not bit for bit — a matmul over more rows need not reproduce one over fewer, and
    /// runners differ in which BLAS kernels they take. Nor rounded to fixed decimals, which
    /// fails whenever two equal-to-rounding values straddle a rounding boundary.</summary>
    private static void Close(double expected, double actual, double rel = 1e-5)
        => Assert.True(Math.Abs(expected - actual) <= rel * Math.Max(1.0, Math.Abs(expected)),
            $"expected {expected:R}, got {actual:R}");

    private const int Horizon = 12;

    private static JointSeries Entry(int variates, int context, bool[] known, int target, int seed)
    {
        var r = new Random(seed);
        var total = context + Horizon + 3;
        var s = new float[variates * total];
        for (var v = 0; v < variates; v++)
        for (var t = 0; t < total; t++) s[v * total + t] = (v + 1) * 10f + MathF.Sin(t / 3f + v) + (float)r.NextDouble();
        return new JointSeries(s, variates, context, known, target);
    }

    private static readonly JointSeries[] Mixed =
    [
        Entry(3, 37, [false, false, true], 1, 1),
        Entry(1, 9, [], 0, 2),
        Entry(2, 70, [false, true], 0, 3),
        Entry(1, 1, [], 0, 4),
        Entry(2, 16, [], 1, 5),
    ];

    [Theory]
    [InlineData(64, true)]
    [InlineData(64, false)]
    [InlineData(3, true)]      // several passes, some entries wider than the budget
    public void JointBatchMatchesSingles(int maxRows, bool averaging)
    {
        var f = InvariantTests.Forecaster();
        f.MaxRowsPerPass = maxRows;
        f.SymmetricAveraging = averaging;
        var q = f.Config.NumQuantiles;
        var batched = new double[f.BatchOutputLength(Mixed.Length, Horizon)];
        f.ForecastJointBatch(Mixed, Horizon, batched);

        for (var i = 0; i < Mixed.Length; i++)
        {
            var e = Mixed[i];
            var single = f.ForecastJoint(e.Series.Span, e.Variates, e.ContextBars, Horizon, e.KnownFuture.Span, e.TargetVariate);
            for (var h = 0; h < Horizon; h++)
            for (var k = 0; k < q; k++)
                Close(single[h, k], batched[(i * Horizon + h) * q + k]);
        }
    }

    [Fact]
    public void OhlcvaBatchMatchesSingles()
    {
        var f = InvariantTests.Forecaster();
        var bars = new float[50 * 6];
        for (var i = 0; i < 50; i++)
        for (var c = 0; c < 6; c++) bars[i * 6 + c] = c < 4 ? 100f + MathF.Sin(i / 4f) + c * 0.1f : 500f + i;
        ReadOnlyMemory<float>[] windows = [bars.AsMemory(0, 20 * 6), bars.AsMemory(6, 33 * 6), bars];
        var batched = new double[f.BatchOutputLength(windows.Length, Horizon)];
        f.ForecastBatch(windows, Horizon, batched);
        var q = f.Config.NumQuantiles;
        for (var i = 0; i < windows.Length; i++)
        {
            var single = f.Forecast(windows[i].Span, Horizon);
            for (var h = 0; h < Horizon; h++)
            for (var k = 0; k < q; k++)
                Close(single[h, k], batched[(i * Horizon + h) * q + k]);
        }
    }

    [Fact]
    public void BadEntriesAndDestinationsAreRefused()
    {
        var f = InvariantTests.Forecaster();
        var size = f.BatchOutputLength(2, Horizon);
        var ex = Assert.Throws<ArgumentException>(() =>
            f.ForecastJointBatch([Mixed[0], Mixed[1] with { KnownFuture = new[] { true } }], Horizon, new double[size]));
        Assert.Contains("batch[1]", ex.Message);
        Assert.Throws<ArgumentException>(() => f.ForecastJointBatch([Mixed[0], Mixed[1]], Horizon, new double[size + 1]));
    }
}

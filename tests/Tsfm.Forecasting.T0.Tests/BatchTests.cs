using TorchSharp;
using Xunit;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.T0.Tests;

/// <summary>
/// Batched calls against the single calls they batch, on a tiny random-init model. Entries
/// are separate groups, so nothing may leak between them; differing contexts are padded
/// to a shared width, which moves rotary positions but not relative ones, so agreement is
/// to float rounding rather than exact.
/// </summary>
public class BatchTests
{
    private static readonly T0Config Tiny = new(
        EmbedDim: 16, NumLayers: 3, NumHeads: 2, MlpHiddenDim: 32, PatchSize: 4, GroupEveryN: 3,
        Quantiles: [0.1, 0.5, 0.9], ScalerUseArcsinh: true, ScalerEps: 0.1, ScalerStdClamp: false,
        XposHalvesLayout: false, ScalerZeroNoValue: true);

    private static T0Forecaster Forecaster(int maxRowsPerPass)
    {
        manual_seed(7);
        var model = new T0Model(Tiny);
        model.eval();
        return new T0Forecaster(model, Tiny, new Device(DeviceType.CPU)) { MaxRowsPerPass = maxRowsPerPass };
    }

    private const int Horizon = 5;

    private static JointSeries Entry(int variates, int context, bool[] known, int target, int seed)
    {
        var r = new Random(seed);
        var total = context + Horizon + 2;                  // spare steps past the horizon are ignored
        var s = new float[variates * total];
        for (var v = 0; v < variates; v++)
        for (var t = 0; t < total; t++) s[v * total + t] = (v + 1) * 10f + MathF.Sin(t / 3f + v) + (float)r.NextDouble();
        return new JointSeries(s, variates, context, known, target);
    }

    private static readonly JointSeries[] Mixed =
    [
        Entry(3, 13, [false, false, true], 1, 1),
        Entry(1, 6, [], 0, 2),
        Entry(4, 21, [false, true, true, false], 3, 3),
        Entry(2, 8, [false, true], 0, 4),
        Entry(1, 1, [], 0, 5),
    ];

    [Theory]
    [InlineData(256)]   // one pass
    [InlineData(4)]     // several passes, one entry wider than the budget
    public void JointBatchMatchesSingles(int maxRowsPerPass)
    {
        var f = Forecaster(maxRowsPerPass);
        var q = Tiny.NumQuantiles;
        var batched = new double[f.BatchOutputLength(Mixed.Length, Horizon)];
        f.ForecastJointBatch(Mixed, Horizon, batched);

        for (var i = 0; i < Mixed.Length; i++)
        {
            var e = Mixed[i];
            var single = f.ForecastJoint(e.Series.Span, e.Variates, e.ContextBars, Horizon, e.KnownFuture.Span, e.TargetVariate);
            for (var h = 0; h < Horizon; h++)
            for (var k = 0; k < q; k++)
                Assert.Equal(single[h, k], batched[(i * Horizon + h) * q + k], 4);
        }
    }

    [Fact]
    public void AnEntryIsUnaffectedByItsCompanions()
    {
        var f = Forecaster(256);
        var size = f.BatchOutputLength(2, Horizon);
        var (a, b) = (new double[size], new double[size]);
        f.ForecastJointBatch([Mixed[0], Mixed[2]], Horizon, a);
        f.ForecastJointBatch([Mixed[0], Entry(4, 21, [false, true, true, false], 3, 99)], Horizon, b);
        var block = Horizon * Tiny.NumQuantiles;
        Assert.Equal(a.AsSpan(0, block).ToArray(), b.AsSpan(0, block).ToArray());
    }

    [Fact]
    public void OhlcvaBatchMatchesSingles()
    {
        var f = Forecaster(256);
        var bars = new float[40 * 6];
        for (var i = 0; i < 40; i++)
        for (var c = 0; c < 6; c++) bars[i * 6 + c] = c < 4 ? 100f + MathF.Sin(i / 4f) + c * 0.1f : 500f + i;

        // Anchors along one history, as a backtest slices them: no copies.
        ReadOnlyMemory<float>[] windows = [bars.AsMemory(0, 20 * 6), bars.AsMemory(6, 25 * 6), bars];
        var batched = new double[f.BatchOutputLength(windows.Length, Horizon)];
        f.ForecastBatch(windows, Horizon, batched);

        var q = Tiny.NumQuantiles;
        for (var i = 0; i < windows.Length; i++)
        {
            var single = f.Forecast(windows[i].Span, Horizon);
            for (var h = 0; h < Horizon; h++)
            for (var k = 0; k < q; k++)
                Assert.Equal(single[h, k], batched[(i * Horizon + h) * q + k], 6);
        }
    }

    [Fact]
    public void BadEntriesAndDestinationsAreRefused()
    {
        var f = Forecaster(256);
        var size = f.BatchOutputLength(2, Horizon);
        var ex = Assert.Throws<ArgumentException>(() =>
            f.ForecastJointBatch([Mixed[0], Mixed[1] with { KnownFuture = new[] { true } }], Horizon, new double[size]));
        Assert.Contains("batch[1]", ex.Message);
        Assert.Throws<ArgumentException>(() => f.ForecastJointBatch([Mixed[0], Mixed[1]], Horizon, new double[size - 1]));
        Assert.Throws<ArgumentException>(() => f.ForecastBatch([new float[6], new float[7]], Horizon, new double[size]));
    }
}

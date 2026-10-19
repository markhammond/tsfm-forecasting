using Xunit;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.TimesFm.IntegrationTests;

/// <summary>
/// Batched calls against the single calls they batch. Run on a small random-init model,
/// so they need no checkpoint and bite in CI: what is under test is the stacking of
/// entries on the batch axis, not the weights.
/// </summary>
public class BatchTests
{
    private static readonly TimesFmConfig Tiny = new(
        InputPatchLen: 8, OutputPatchLen: 16, Quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        NumLayers: 2, ModelDims: 32, NumHeads: 4, UseVariateAttention: true, MaxVariates: 8, ValueClip: 1000);

    private static TimesFmForecaster Forecaster(int maxBatchPerPass)
    {
        manual_seed(11);
        var model = new TimesFmModel(Tiny);
        model.eval();
        return new TimesFmForecaster(model, Tiny, new Device("cpu")) { MaxBatchPerPass = maxBatchPerPass };
    }

    private static JointSeries Entry(int variates, int total, int context, bool[] known, int target, int seed)
    {
        var r = new Random(seed);
        var s = new float[variates * total];
        for (var i = 0; i < s.Length; i++) s[i] = 20f * (i / total + 1) + (float)r.NextDouble();
        return new JointSeries(s, variates, context, known, target);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(2)]
    public void JointBatchMatchesSinglesExactly(int maxBatchPerPass)
    {
        // Two shapes, interleaved; one pass per shape, or several when capped.
        JointSeries[] batch =
        [
            Entry(3, 32, 16, [false, false, true], 0, 1),
            Entry(2, 24, 8, [], 1, 2),
            Entry(3, 32, 16, [false, true, true], 0, 3),
            Entry(3, 32, 16, [], 2, 4),
            Entry(2, 24, 8, [false, true], 0, 5),
        ];
        const int horizon = 6;
        var f = Forecaster(maxBatchPerPass);
        var q = Tiny.Quantiles.Length;
        var batched = new double[f.BatchOutputLength(batch.Length, horizon)];
        f.ForecastJointBatch(batch, horizon, batched);

        for (var i = 0; i < batch.Length; i++)
        {
            var e = batch[i];
            var single = f.ForecastJoint(e.Series.Span, e.Variates, e.ContextBars, horizon, e.KnownFuture.Span, e.TargetVariate);
            for (var h = 0; h < horizon; h++)
            for (var k = 0; k < q; k++)
                Assert.Equal(single[h, k], batched[(i * horizon + h) * q + k]);
        }
    }

    [Fact]
    public void OhlcvaBatchMatchesSinglesExactly()
    {
        var bars = new float[32 * 6];
        for (var i = 0; i < bars.Length; i++) bars[i] = 100f + MathF.Sin(i / 7f);
        ReadOnlyMemory<float>[] windows = [bars.AsMemory(0, 16 * 6), bars.AsMemory(16 * 6), bars, bars.AsMemory(8 * 6, 16 * 6)];
        const int horizon = 5;

        var f = Forecaster(64);
        var q = Tiny.Quantiles.Length;
        var batched = new double[f.BatchOutputLength(windows.Length, horizon)];
        f.ForecastBatch(windows, horizon, batched);
        for (var i = 0; i < windows.Length; i++)
        {
            var single = f.Forecast(windows[i].Span, horizon);
            for (var h = 0; h < horizon; h++)
            for (var k = 0; k < q; k++)
                Assert.Equal(single[h, k], batched[(i * horizon + h) * q + k]);
        }
    }
}

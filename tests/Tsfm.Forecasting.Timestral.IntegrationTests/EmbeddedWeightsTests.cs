using Tsfm.Forecasting.Timestral.Weights;

using Xunit;
using Xunit.Abstractions;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.Timestral.IntegrationTests;

/// <summary>
/// The embedded 5M and 28M checkpoints, end to end. Numerical agreement with the reference
/// is the parity harness's job; these check that the packages carry what they claim and
/// that the forecaster behaves on real weights.
/// </summary>
public class EmbeddedWeightsTests(ITestOutputHelper output)
{
    public static TheoryData<IModelWeights> Checkpoints => [Timestral5M.Instance, Timestral28M.Instance];

    private const int Context = 96, Horizon = 24, Total = Context + Horizon;

    private static float[] Series(float driverFuture)
    {
        var s = new float[3 * Total];
        for (var t = 0; t < Total; t++)
        {
            s[Total + t] = t < Context ? 20f + 4f * MathF.Sin(t / 6f) : 20f + driverFuture;
            s[2 * Total + t] = t % 7 < 5 ? 1f : 0f;
        }
        for (var t = 0; t < Context; t++) s[t] = 100f + 3f * s[Total + Math.Max(t - 1, 0)] + s[2 * Total + t];
        return s;
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void LoadsWithProvenanceAndTheTrainedGrid(IModelWeights weights)
    {
        var f = TimestralForecaster.Load(weights, new Device("cpu"));
        Assert.Contains("@", f.CheckpointName);
        Assert.Equal(21, f.Quantiles.Count);
        Assert.Equal(1024, f.MaxHorizon);
        Assert.Equal(32768, f.MaxContext);
    }

    [Fact]
    public void TheLicenceTravelsWithTheWeights()
    {
        Assert.Contains("Modified MIT License", Timestral5M.License);
        Assert.Contains("$20 million", Timestral28M.License);
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void KnownFutureValuesLiftTheForecast(IModelWeights weights)
    {
        var f = TimestralForecaster.Load(weights, new Device("cpu"));
        var a = f.ForecastJoint(Series(+8f), 3, Context, Horizon, [false, true, true]);
        var b = f.ForecastJoint(Series(-8f), 3, Context, Horizon, [false, true, true]);
        var mid = f.Config.MedianIndex;
        var lift = Enumerable.Range(0, Horizon).Average(h => a[h, mid] - b[h, mid]);
        output.WriteLine($"{weights.Name}: mean median lift {lift:F2}");
        Assert.True(lift > 0, "a target that tracks its driver should rise with the driver's known future");
    }

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void ForecastsAreOrderedAndFinite(IModelWeights weights)
    {
        var f = TimestralForecaster.Load(weights, new Device("cpu"));
        var q = f.ForecastJoint(Series(0f), 3, Context, Horizon, [false, true, true]);
        for (var h = 0; h < Horizon; h++)
        for (var k = 0; k < q.GetLength(1); k++)
        {
            Assert.True(double.IsFinite(q[h, k]));
            if (k > 0) Assert.True(q[h, k] >= q[h, k - 1]);
        }
    }
}

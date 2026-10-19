using TorchSharp;
using Xunit;
using Xunit.Abstractions;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.T0.Tests;

/// <summary>
/// Behaviour of the published checkpoints through <see cref="T0Forecaster.ForecastJoint"/>.
///
/// <para>Gated on the checkpoint being present, and so vacuous in CI, which does not fetch
/// ~1.4 GB of weights. Fetch them with scripts/fetch-t0-checkpoint.sh to run these locally;
/// numerical agreement with the reference is the parity harness's job, not this one's.</para>
/// </summary>
public class CheckpointTests(ITestOutputHelper output)
{
    private const int Context = 96, Horizon = 24, Total = Context + Horizon, Variates = 3;

    private static string Dir(string name) => Path.Combine(
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../..")), "checkpoints", name);

    private T0Forecaster? Load(string name)
    {
        if (File.Exists(Path.Combine(Dir(name), "model.safetensors")))
            return T0Forecaster.Load(Dir(name), new Device(DeviceType.CPU));
        output.WriteLine($"{name} absent — vacuous");
        return null;
    }

    /// <summary>Variate 0 is the target, driven by variate 1 one step ahead; 1 is known
    /// into the future; 2 is a calendar flag.</summary>
    private static float[] Series(float driverFuture)
    {
        var s = new float[Variates * Total];
        for (var t = 0; t < Total; t++)
        {
            var driver = t < Context ? 20f + 4f * MathF.Sin(t / 6f) : 20f + driverFuture;
            s[1 * Total + t] = driver;
            s[2 * Total + t] = t % 7 < 5 ? 1f : 0f;
        }
        for (var t = 0; t < Context; t++)
            s[t] = 100f + 3f * s[Total + Math.Max(t - 1, 0)] + s[2 * Total + t];
        return s;
    }

    [Theory]
    [InlineData("t0-alpha", 5)]
    [InlineData("t0-beta", 21)]
    public void QuantileGridComesFromTheCheckpoint(string name, int levels)
    {
        var f = Load(name); if (f is null) return;
        Assert.Equal(levels, f.Quantiles.Count);
        Assert.Contains(0.5, f.Quantiles);
    }

    [Theory]
    [InlineData("t0-alpha")]
    [InlineData("t0-beta")]
    public void KnownFutureValuesChangeTheForecast(string name)
    {
        var f = Load(name); if (f is null) return;
        var a = f.ForecastJoint(Series(+8f), Variates, Context, Horizon, [false, true, true]);
        var b = f.ForecastJoint(Series(-8f), Variates, Context, Horizon, [false, true, true]);

        var mid = Array.IndexOf(f.Config.Quantiles, 0.5);
        var lift = Enumerable.Range(0, Horizon).Average(h => a[h, mid] - b[h, mid]);
        output.WriteLine($"{name}: mean median lift from a higher driver = {lift:F2}");
        Assert.True(lift > 0, "a target that tracks its driver should rise with the driver's known future");
    }

    [Theory]
    [InlineData("t0-alpha")]
    [InlineData("t0-beta")]
    public void IdenticalInputGivesIdenticalForecast(string name)
    {
        var f = Load(name); if (f is null) return;
        var s = Series(0f);
        Assert.Equal(f.ForecastJoint(s, Variates, Context, Horizon, [false, true, true]),
                     f.ForecastJoint(s, Variates, Context, Horizon, [false, true, true]));
    }

    [Theory]
    [InlineData("t0-alpha")]
    [InlineData("t0-beta")]
    public void QuantilesAreOrderedAndFinite(string name)
    {
        var f = Load(name); if (f is null) return;
        var q = f.ForecastJoint(Series(0f), Variates, Context, Horizon, [false, true, true]);
        for (var h = 0; h < Horizon; h++)
        for (var i = 0; i < q.GetLength(1); i++)
        {
            Assert.True(double.IsFinite(q[h, i]));
            if (i > 0) Assert.True(q[h, i] >= q[h, i - 1]);
        }
    }
}

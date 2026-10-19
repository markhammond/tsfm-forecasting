using TorchSharp;
using Xunit;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.Timestral.Tests;

/// <summary>
/// Properties that hold for any weights, on a tiny random-init model — the reference's own
/// test configuration — so they run in CI without a checkpoint.
/// </summary>
public class InvariantTests
{
    internal static readonly TimestralConfig Tiny = new(
        PatchSize: 8, DModel: 32, NumHeads: 4, MlpHiddenDim: 64,
        BlockSchedule: ["time", "time_variate", "time", "time_variate"],
        Quantiles: [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95],
        BlockSize: 4, ContextLength: 256, RopeTheta: 10000, RopePartialFactor: 0.5, NormEps: 1e-6,
        AttnBias: false, MlpBias: false, TimeAttnGate: true, VariateAttnGate: true, MinimumScale: 1e-6,
        UseAsinh: true, CovariateEnabled: true, SplitContextHorizonAttn: true, AttendMissingPatches: true,
        QuantileWidthFactor: 3.0);

    internal static TimestralForecaster Forecaster(bool raw = false)
    {
        manual_seed(0);
        var model = new TimestralModel(Tiny);
        using (no_grad())
            foreach (var (_, p) in model.named_parameters()) p.normal_(0, 0.05);
        model.eval();
        return new TimestralForecaster(model, new Device(DeviceType.CPU), "tiny")
            { SymmetricAveraging = !raw, ClipNonNegative = !raw };
    }

    private const int Context = 37, Horizon = 20, Total = Context + Horizon;
    private static readonly bool[] Known = [false, false, true];

    internal static float[] Series(Func<int, int, float>? tweak = null)
    {
        var s = new float[3 * Total];
        for (var t = 0; t < Total; t++)
        {
            s[t] = 50f + 3f * MathF.Sin(t / 2f);
            s[Total + t] = -2f * MathF.Cos(t / 3f);
            s[2 * Total + t] = 10f + t % 4;
        }
        if (tweak is not null)
            for (var v = 0; v < 3; v++)
            for (var t = 0; t < Total; t++) s[v * Total + t] += tweak(v, t);
        return s;
    }

    [Fact]
    public void TargetFutureRegionIsNeverRead()
    {
        var f = Forecaster();
        Assert.Equal(f.ForecastJoint(Series(), 3, Context, Horizon, Known),
            f.ForecastJoint(Series((v, t) => v < 2 && t >= Context ? 1e4f : 0f), 3, Context, Horizon, Known));
    }

    [Fact]
    public void KnownFutureValuesReachTheTarget()
    {
        var f = Forecaster();
        Assert.NotEqual(f.ForecastJoint(Series(), 3, Context, Horizon, Known),
            f.ForecastJoint(Series((v, t) => v == 2 && t >= Context ? 5f : 0f), 3, Context, Horizon, Known));
    }

    /// <summary>Covariates are exogenous: perturbing a target's context must not move a
    /// known-future row's output.</summary>
    [Fact]
    public void KnownFutureRowsDoNotReadTargets()
    {
        var f = Forecaster(raw: true);
        using var _ = no_grad();
        const int ctxPatches = 4, horizonPatches = 2, p = 8;
        const int w = (ctxPatches + horizonPatches) * p;
        Tensor Known(float level)
        {
            var values = zeros(2, w);
            values[0, TensorIndex.Slice(0, ctxPatches * p)] = arange(ctxPatches * p, ScalarType.Float32) * 0.1f + level;
            values[1] = arange(w, ScalarType.Float32).sin();
            var observed = ones([2, w], ScalarType.Bool);
            observed[0, TensorIndex.Slice(ctxPatches * p)] = false;
            var roles = tensor(new long[] { Role.Target, Role.KnownCovariate });
            return f.Model.PredictBlock(values, observed, zeros(2, ScalarType.Int64), roles, ctxPatches, horizonPatches)[1];
        }
        Assert.True(Known(0f).allclose(Known(7f)));
    }

    /// <summary>Negating every input reflects the averaged forecast exactly — the defining
    /// property of the reference's symmetric averaging.</summary>
    [Fact]
    public void SymmetricAveragingIsOdd()
    {
        var f = Forecaster();
        f.ClipNonNegative = false;
        var a = f.ForecastJoint(Series(), 3, Context, Horizon, Known);
        var b = f.ForecastJoint(Series().Select(x => -x).ToArray(), 3, Context, Horizon, Known);
        var q = a.GetLength(1);
        for (var h = 0; h < Horizon; h++)
        for (var k = 0; k < q; k++)
            Assert.Equal(a[h, k], -b[h, q - 1 - k], 4);
    }

    [Fact]
    public void NonNegativeHistoryIsClippedAtZero()
    {
        var f = Forecaster();
        var s = Enumerable.Range(0, 40).Select(t => t % 5 == 0 ? 0f : 0.01f).ToArray();
        Assert.All(f.ForecastJoint(s, 1, 30, 10).Cast<double>(), x => Assert.True(x >= 0));
    }

    [Fact]
    public void AnyContextLengthAndMissingValuesAreAccepted()
    {
        var f = Forecaster();
        for (var context = 1; context <= 17; context += 4)
        {
            var s = Enumerable.Range(0, context + 3).Select(t => t == 0 && context > 2 ? float.NaN : 10f + t).ToArray();
            Assert.All(f.ForecastJoint(s, 1, context, 3).Cast<double>(), x => Assert.True(double.IsFinite(x)));
        }
    }

    [Fact]
    public void QuantilesAreOrdered()
    {
        var q = Forecaster().ForecastJoint(Series(), 3, Context, Horizon, Known);
        for (var h = 0; h < Horizon; h++)
        for (var k = 1; k < q.GetLength(1); k++)
            Assert.True(q[h, k] >= q[h, k - 1]);
    }

    [Fact]
    public void ArgumentsAreValidated()
    {
        var f = Forecaster();
        var s = Series();
        Assert.Throws<ArgumentException>(() => f.ForecastJoint(s, 3, Context, Horizon, [true, false, true]));
        Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(s, 3, Context, f.MaxHorizon + 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(s, 3, Context, Horizon + 1, Known));
        var tooLong = new float[Tiny.ContextLength + 1];
        Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(tooLong, 1, tooLong.Length, 1));
        Assert.Throws<ArgumentException>(() => f.Forecast(new float[7], 1));
    }

    [Fact]
    public void ConfigRefusesUnknownKeys()
    {
        const string json = """{"patch_size":32,"d_model":192,"num_heads":3,"mlp_hidden_dim":512,"block_schedule":["time","time_variate"],"quantile_knots":[0.1,0.5,0.9],"block_size":32,"context_length":32768,"quantile_width_factor":3.0}""";
        Assert.Equal(3.0, TimestralConfig.Parse(json).QuantileWidthFactor);
        Assert.Throws<InvalidDataException>(() => TimestralConfig.Parse(json.Replace("\"block_size\"", "\"new_knob\":1,\"block_size\"")));
        Assert.Throws<InvalidDataException>(() => TimestralConfig.Parse(json.Replace("[0.1,0.5,0.9]", "[0.1,0.9]")));
    }
}

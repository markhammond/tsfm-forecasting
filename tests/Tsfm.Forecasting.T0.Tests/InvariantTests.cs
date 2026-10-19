using TorchSharp;
using Xunit;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.T0.Tests;

/// <summary>
/// Structural properties that hold for any weights, checked on a tiny randomly initialised
/// model so they run in CI without a checkpoint. Each is a leak or a blind spot that would
/// otherwise forecast plausibly: a target reading its own future, a known-future row
/// reading a target, padding acting as data.
/// </summary>
public class InvariantTests
{
    private static readonly T0Config Tiny = new(
        EmbedDim: 16, NumLayers: 3, NumHeads: 2, MlpHiddenDim: 32, PatchSize: 4, GroupEveryN: 3,
        Quantiles: [0.1, 0.5, 0.9], ScalerUseArcsinh: true, ScalerEps: 0.1, ScalerStdClamp: false,
        XposHalvesLayout: false, ScalerZeroNoValue: true);

    private static T0Forecaster Forecaster()
    {
        manual_seed(7);
        var model = new T0Model(Tiny);
        model.eval();
        return new T0Forecaster(model, Tiny, new Device(DeviceType.CPU));
    }

    private const int Context = 10, Horizon = 6, Total = Context + Horizon;

    /// <summary>Variate 0 a target, 1 a second target, 2 known into the future.</summary>
    private static float[] Series(Func<int, int, float>? tweak = null)
    {
        var s = new float[3 * Total];
        for (var t = 0; t < Total; t++)
        {
            s[0 * Total + t] = 50f + 3f * MathF.Sin(t / 2f);
            s[1 * Total + t] = 2f * MathF.Cos(t / 3f);
            s[2 * Total + t] = t % 4;
        }
        if (tweak is not null)
            for (var v = 0; v < 3; v++)
            for (var t = 0; t < Total; t++)
                s[v * Total + t] += tweak(v, t);
        return s;
    }

    private static readonly bool[] Known = [false, false, true];

    [Fact]
    public void TargetFutureRegionIsNeverRead()
    {
        var f = Forecaster();
        var a = f.ForecastJoint(Series(), 3, Context, Horizon, Known);
        var b = f.ForecastJoint(Series((v, t) => v < 2 && t >= Context ? 1e4f : 0f), 3, Context, Horizon, Known);
        Assert.Equal(a, b);
    }

    [Fact]
    public void KnownFutureValuesReachTheTarget()
    {
        var f = Forecaster();
        var a = f.ForecastJoint(Series(), 3, Context, Horizon, Known);
        var b = f.ForecastJoint(Series((v, t) => v == 2 && t >= Context ? 5f : 0f), 3, Context, Horizon, Known);
        Assert.NotEqual(a, b);
    }

    /// <summary>The group mask lets known-future rows read only each other. If it leaked,
    /// a known row's representation — and through it the target's — would depend on the
    /// target's history in both directions.</summary>
    [Fact]
    public void KnownFutureRowsDoNotReadTargets()
    {
        manual_seed(7);
        using var model = new T0Model(Tiny);
        model.eval();
        using var _ = no_grad();

        // Rows: target, known-future. Width 16 = 4 patches; context 2 patches, horizon 2.
        const int w = 16, ctx = 8;
        long[] dims = [2, w];
        var mask = zeros(dims, ScalarType.Int64);
        mask[0, TensorIndex.Slice(ctx)] = MaskType.Withheld;
        var groups = zeros(dims, ScalarType.Int64);
        var types = zeros(dims, ScalarType.Int64);
        types[1] = VariateType.Future;

        Tensor KnownRowOutput(float targetLevel)
        {
            var values = stack([full(w, targetLevel), arange(w, ScalarType.Float32)]);
            var (scaled, _, _) = T0Preprocess.ScaleInput(values, mask, groups, types, Tiny);
            // Perturb in scaled space so the scaler cannot absorb it.
            scaled[0, TensorIndex.Slice(0, ctx)] = scaled[0, TensorIndex.Slice(0, ctx)] + targetLevel;
            return model.Backbone(scaled, mask, groups, types)[1];
        }

        Assert.True(KnownRowOutput(0f).allclose(KnownRowOutput(3f)));
    }

    /// <summary>Time attention is causal for targets: an output must not move when only
    /// later steps change.</summary>
    [Fact]
    public void TargetOutputsAreCausalInTime()
    {
        manual_seed(7);
        using var model = new T0Model(Tiny);
        model.eval();
        using var _ = no_grad();

        const int w = 16;
        long[] dims = [1, w];
        var (mask, groups, types) = (zeros(dims, ScalarType.Int64), zeros(dims, ScalarType.Int64),
            zeros(dims, ScalarType.Int64));
        var a = randn(dims);
        var b = a.clone();
        b[0, TensorIndex.Slice(12)] = b[0, TensorIndex.Slice(12)] + 10f;    // last patch only

        var qa = model.Backbone(a, mask, groups, types);
        var qb = model.Backbone(b, mask, groups, types);
        Assert.True(qa[0, TensorIndex.Slice(0, 3)].allclose(qb[0, TensorIndex.Slice(0, 3)]));
        Assert.False(qa[0, 3].allclose(qb[0, 3]));
    }

    /// <summary>A context that does not fill whole patches is left-padded, and padding must
    /// be inert: whatever the pad holds, it is masked from attention and scaling alike.</summary>
    [Fact]
    public void AnyContextLengthIsAccepted()
    {
        var f = Forecaster();
        for (var context = 1; context <= 9; context++)
        {
            var series = Enumerable.Range(0, context + 3).Select(t => 10f + t).ToArray();
            var q = f.ForecastJoint(series, 1, context, 3);
            Assert.Equal(3, q.GetLength(0));
            Assert.All(q.Cast<double>(), x => Assert.True(double.IsFinite(x)));
        }
    }

    [Fact]
    public void QuantilesAreMonotoneAndFinite()
    {
        var f = Forecaster();
        var q = f.ForecastJoint(Series(), 3, Context, Horizon, Known);
        for (var h = 0; h < Horizon; h++)
        for (var i = 0; i < q.GetLength(1); i++)
        {
            Assert.True(double.IsFinite(q[h, i]));
            if (i > 0) Assert.True(q[h, i] >= q[h, i - 1]);
        }
    }

    [Fact]
    public void MissingObservationsAreTolerated()
    {
        var f = Forecaster();
        var s = Series();
        s[3] = float.NaN;
        s[2 * Total + Context + 1] = float.NaN;       // a gap in the known future
        Assert.All(f.ForecastJoint(s, 3, Context, Horizon, Known).Cast<double>(),
            x => Assert.True(double.IsFinite(x)));
    }

    [Fact]
    public void OhlcvaOfAnyLengthForecastsReturns()
    {
        var f = Forecaster();
        var bars = new float[13 * 6];
        for (var i = 0; i < 13; i++)
        for (var c = 0; c < 6; c++) bars[i * 6 + c] = c < 4 ? 100f + i : 1000f;
        var q = f.Forecast(bars, 5);
        Assert.Equal(5, q.GetLength(0));
        Assert.Equal(Tiny.NumQuantiles, q.GetLength(1));
    }

    [Fact]
    public void TheForecasterIsAnIQuantileForecaster()
    {
        IQuantileForecaster f = Forecaster();
        Assert.Equal(Tiny.Quantiles, f.Quantiles);
    }

    [Fact]
    public void ArgumentsAreValidated()
    {
        var f = Forecaster();
        var s = Series();
        Assert.Throws<ArgumentException>(() => f.ForecastJoint(s, 3, Context, Horizon, [true, false, true]));
        Assert.Throws<ArgumentException>(() => f.ForecastJoint(s, 3, Context, Horizon, [false, true]));
        Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(s, 3, Context, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(s, 3, Context, T0Forecaster.MaxHorizon + 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(s, 3, Context, Horizon + 1, Known));
        Assert.Throws<ArgumentOutOfRangeException>(() => f.ForecastJoint(s, 3, 0, Horizon));
        Assert.Throws<ArgumentException>(() => f.ForecastJoint(s, 5, Context, Horizon));
        Assert.Throws<ArgumentException>(() => f.Forecast(new float[7], 1));
    }

    [Fact]
    public void ConfigReadsBetaConventionsAndDefaultsToAlphas()
    {
        const string alpha = """{"embed_dim":512,"group_every_n":3,"mlp_hidden_dim":2048,"num_heads":8,"num_layers":24,"patch_size":32,"quantile_levels":[0.1,0.25,0.5,0.75,0.9],"scaler_use_arcsinh":true}""";
        const string beta = """{"embed_dim":1024,"group_every_n":3,"mlp_hidden_dim":2048,"num_heads":8,"num_layers":24,"patch_size":32,"quantile_levels":[0.1,0.5,0.9],"scaler_eps":0.01,"scaler_eps_mode":"std_clamp","scaler_use_arcsinh":true}""";

        var a = T0Config.Parse(alpha);
        Assert.Equal(0.1, a.ScalerEps);
        Assert.False(a.ScalerStdClamp);
        Assert.False(a.XposHalvesLayout);
        Assert.True(a.ScalerZeroNoValue);

        var b = T0Config.Parse(beta);
        Assert.Equal(0.01, b.ScalerEps);
        Assert.True(b.ScalerStdClamp);

        Assert.Throws<InvalidDataException>(() => T0Config.Parse(beta.Replace("std_clamp", "bogus")));
    }
}

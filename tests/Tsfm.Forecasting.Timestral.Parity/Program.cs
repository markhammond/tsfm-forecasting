// Bisecting parity harness for Timestral. Compares the .NET port against tensors dumped
// from mistralai/timestral by reference/dump_timestral_parity.py, one stage at a time, then
// end to end — with the reference forecaster's defaults and without them.
//
// Each stage is fed the REFERENCE output of the stage before it, so a failure localises to
// one stage rather than being inherited from an earlier one.
//
//   dotnet run --project tests/Tsfm.Forecasting.Timestral.Parity -c Release -- Timestral-5M Timestral-28M

using TorchSharp;

using Tsfm.Forecasting;
using Tsfm.Forecasting.Timestral;

using static TorchSharp.torch;

var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
var names = args.Length > 0 ? args : ["Timestral-5M"];

Device device;
try { device = new Device(DeviceType.MPS); using (ones(1).to(device)) { } }
catch { device = new Device(DeviceType.CPU); }
Console.WriteLine($"device={device.type}");

var ok = true;
foreach (var name in names) ok &= Run(name);
Console.WriteLine();
Console.WriteLine(ok ? "PARITY: PASS" : "PARITY: FAIL — the first failing stage above localises the fault");
return ok ? 0 : 1;

bool Run(string name)
{
    var fixturePath = Path.Combine(root, "reference", $"{name}-parity.safetensors");
    if (!File.Exists(fixturePath))
    {
        Console.Error.WriteLine($"missing {fixturePath}; run reference/dump_timestral_parity.py checkpoints/{name}");
        return false;
    }

    using var _ = no_grad();
    using var fx = Safetensors.Load(fixturePath);
    Tensor E(string n) => fx.Get(n).to(device);

    var forecaster = TimestralForecaster.Load(Path.Combine(root, "checkpoints", name), device);
    var (model, c) = (forecaster.Model, forecaster.Config);
    var pass = true;

    Console.WriteLine($"\n── {name}: {c.BlockSchedule.Length} layers x {c.DModel}, {c.NumHeads} heads, rope {c.RopeDim}/{c.HeadDim}");
    Console.WriteLine($"{"stage",-24} {"max|Δ|",12} {"rel",12}  note");

    void Check(string stage, Tensor got, Tensor want, double tol, string note = "")
    {
        var absMax = (got.to(ScalarType.Float32).to(want.device) - want).abs().max().item<float>();
        var scale = want.abs().max().item<float>();
        var rel = absMax / Math.Max(scale, 1e-12);
        var good = rel <= tol || absMax <= 1e-6;
        pass &= good;
        Console.WriteLine($"{stage,-24} {absMax,12:E3} {rel,12:E3}  {(good ? "ok" : "FAIL")} {note}");
    }

    // 1. Scaling: causal for targets, instance statistics for known-future rows.
    var values = E("buf_values");
    var observed = E("buf_observed").to(ScalarType.Bool);
    var groups = E("buf_groups").to(ScalarType.Int64);
    var roles = E("buf_roles").to(ScalarType.Int64);
    var ll = (int)fx.Get("ctx_patches").item<float>();
    var p = c.PatchSize;
    var (r, w) = (values.shape[0], (int)(values.shape[1] / p));
    var h = w - ll;
    var (scaled, loc, scale) = TimestralPreprocess.Scale(values, observed, roles, c);
    Check("loc", loc, E("loc"), 1e-6);
    Check("scale", scale, E("scale"), 1e-6);
    Check("scaled", scaled, E("scaled"), 1e-5);

    // 2. Embedding, masks and positions.
    Check("patch_embedding", model.representation.Embed(E("scaled"), observed), E("patch_embedding"), 1e-5);
    var perPatch = observed.reshape(r, w, p).any(-1);
    var keyValid = TimestralPreprocess.KeyValid(perPatch);
    var timeMask = TimestralPreprocess.TimeMask(keyValid, roles.eq(Role.KnownCovariate), ll, h);
    var variateMask = TimestralPreprocess.VariateMask(groups, keyValid, roles, h);
    var positions = TimestralPreprocess.Positions(TimestralPreprocess.FirstObserved(perPatch, groups), w, ll, h);
    Check("time_mask", timeMask.to(ScalarType.Float32), E("time_mask"), 0);
    Check("variate_mask", variateMask.to(ScalarType.Float32), E("variate_mask"), 0);
    Check("positions", positions.to(ScalarType.Float32), E("positions"), 0);

    // 3. Layers, each from the reference's previous output.
    var tf = model.transformer;
    var x0 = cat([E("patch_embedding"), model.representation.ForecastTokens(r, h)], 1);
    var x = x0;
    var rope = new PartialRope(c.HeadDim, c.RopeDim, c.RopeTheta).Tables(E("positions").to(ScalarType.Int64));
    for (var i = 0; i < tf.layers.Count; i++)
    {
        x = tf.layers[i].Forward(x, timeMask, variateMask, rope.Cos, rope.Sin, w);
        var refName = $"layer{i}_output";
        if (!fx.Has(refName)) continue;
        Check(refName, x, E(refName), 1e-4, c.BlockSchedule[i]);
        x = E(refName);
    }
    Check("transformer_output", tf.Forward(x0, timeMask, variateMask, positions, w), E("transformer_output"), 2e-4,
        "(whole stack)");

    // 4. Head, then the whole block end to end.
    Check("head_output", model.head.Forward(E("transformer_output").narrow(1, w, h)), E("head_output"), 1e-5);
    Check("block_output", model.PredictBlock(values, observed, groups, roles, ll, h), E("block_output"), 1e-4);

    // 5. End to end through the public API.
    var ctx = fx.Get("in_ctx");                     // (2, T)
    var fut = fx.Get("in_fut");                     // (2, T + H)
    var (t, horizon) = ((int)ctx.shape[1], (int)(fut.shape[1] - ctx.shape[1]));
    var total = t + horizon;
    var joint = new float[4 * total];
    for (var v = 0; v < 2; v++)
    for (var i = 0; i < t; i++) joint[v * total + i] = ctx[v, i].item<float>();
    for (var v = 0; v < 2; v++)
    for (var i = 0; i < total; i++) joint[(2 + v) * total + i] = fut[v, i].item<float>();

    for (var target = 0; target < 2; target++)
    {
        Check($"joint[{target}]", Grid(forecaster.ForecastJoint(joint, 4, t, horizon, [false, false, true, true], target)),
            E("predict_joint")[target], 1e-4, "(defaults)");
        forecaster.SymmetricAveraging = forecaster.ClipNonNegative = false;
        Check($"joint_raw[{target}]", Grid(forecaster.ForecastJoint(joint, 4, t, horizon, [false, false, true, true], target)),
            E("predict_joint_raw")[target], 1e-4, "(no averaging, no clip)");
        forecaster.SymmetricAveraging = forecaster.ClipNonNegative = true;
        Check($"targets_only[{target}]", Grid(forecaster.ForecastJoint(joint.AsSpan(0, 2 * total), 2, t, horizon,
            targetVariate: target)), E("predict_targets_only")[target], 1e-4);
    }
    var aligned = Floats(ctx[0].narrow(0, 64, 64));
    Check("aligned", Grid(forecaster.ForecastJoint(aligned, 1, 64, 32)), E("predict_aligned"), 1e-4, "(no padding)");

    // 6. The quickstart notebooks' series, shared with the t0 harness.
    var m4 = Floats(fx.Get("m4_ctx"));
    Check("notebook m4 H1", Grid(forecaster.ForecastJoint(m4, 1, m4.Length, 48)), E("m4_predict"), 1e-4);
    var ep = Floats(fx.Get("ep_ctx"));
    var cov = fx.Get("ep_cov");
    var truth = Floats(fx.Get("ep_truth"));
    var plain = forecaster.ForecastJoint(ep, 1, ep.Length, truth.Length);
    Check("notebook ep", Grid(plain), E("ep_predict"), 1e-4);
    var width = (int)cov.shape[1];
    var epJoint = new float[3 * width];
    ep.CopyTo(epJoint, 0);
    Floats(cov[0]).CopyTo(epJoint, width);
    Floats(cov[1]).CopyTo(epJoint, 2 * width);
    var withCov = forecaster.ForecastJoint(epJoint, 3, ep.Length, truth.Length, [false, true, true]);
    Check("notebook ep + cov", Grid(withCov), E("ep_predict_cov"), 1e-4);

    // 7. One batch of mixed lengths and kinds — including a known-future entry, which
    // widens the whole pass's encoder over the horizon — against the single forecasts.
    // All share the electricity horizon of 24; the refs were taken at 40, 32, 48 and 24, and
    // only target-only entries truncate exactly (their later patches cannot reach earlier).
    const int shared = 24;
    JointSeries[] batch =
    [
        new(joint.AsMemory(0, 2 * total), 2, t, TargetVariate: 1),
        new(aligned, 1, 64),
        new(m4, 1, m4.Length),
        new(epJoint, 3, ep.Length, new[] { false, true, true }),
    ];
    var outBatch = new double[forecaster.BatchOutputLength(batch.Length, shared)];
    forecaster.ForecastJointBatch(batch, shared, outBatch);
    for (var i = 0; i < batch.Length; i++)
    {
        var e = batch[i];
        var single = Grid(forecaster.ForecastJoint(e.Series.Span, e.Variates, e.ContextBars, shared,
            e.KnownFuture.Span, e.TargetVariate));
        var got = tensor(outBatch.AsSpan(i * shared * c.NumQuantiles, shared * c.NumQuantiles).ToArray()
            .Select(d => (float)d).ToArray(), [shared, c.NumQuantiles]);
        Check($"batch[{i}]", got, single.to(device), 1e-4, "(vs single call)");
    }

    var mid = c.MedianIndex;
    Console.WriteLine($"  electricity price |median-truth|: {MeanAbs(plain, truth, mid):F4} without covariates, "
                      + $"{MeanAbs(withCov, truth, mid):F4} with");
    return pass;
}

static float[] Floats(Tensor x) => x.contiguous().data<float>().ToArray();

static Tensor Grid(double[,] q) => tensor(q.Cast<double>().Select(d => (float)d).ToArray(),
    [q.GetLength(0), q.GetLength(1)]);

static double MeanAbs(double[,] q, float[] truth, int col)
    => Enumerable.Range(0, truth.Length).Average(h => Math.Abs(q[h, col] - truth[h]));

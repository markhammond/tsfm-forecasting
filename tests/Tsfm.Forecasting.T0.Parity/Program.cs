// Bisecting parity harness for t0. Compares the .NET port against tensors dumped from the
// published tfc-t0 package by reference/dump_t0_parity.py, one stage at a time, then end
// to end through ForecastJoint — including the quickstart notebook's own examples.
//
// Each stage is fed the REFERENCE output of the stage before it, so a failure localises to
// one stage rather than being inherited from an earlier one.
//
//   dotnet run --project tests/Tsfm.Forecasting.T0.Parity -c Release -- t0-alpha t0-beta

using TorchSharp;

using Tsfm.Forecasting;
using Tsfm.Forecasting.T0;

using static TorchSharp.torch;

var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
var names = args.Length > 0 ? args : ["t0-alpha"];

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
        Console.Error.WriteLine($"missing {fixturePath}; run reference/dump_t0_parity.py checkpoints/{name}");
        return false;
    }

    using var _ = no_grad();
    using var fx = Safetensors.Load(fixturePath);
    Tensor E(string n) => fx.Get(n).to(device);
    Tensor L(string n) => E(n).to(ScalarType.Int64);

    var forecaster = T0Forecaster.Load(Path.Combine(root, "checkpoints", name), device);
    var (model, cfg) = (forecaster.Model, forecaster.Config);
    var pass = true;

    Console.WriteLine($"\n── {name}: {cfg.NumLayers} layers x {cfg.EmbedDim}, {cfg.NumQuantiles} quantiles, "
                      + $"eps {cfg.ScalerEps} {(cfg.ScalerStdClamp ? "std_clamp" : "variance_offset")}");
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

    // 1. Scaling: causal statistics for targets, global for known-future rows.
    var (values, mask, groups, types) = (E("buf_values"), L("buf_mask"), L("buf_groups"), L("buf_types"));
    var (scaled, loc, scale) = T0Preprocess.ScaleInput(values, mask, groups, types, cfg);
    Check("loc", loc, E("loc"), 1e-6);
    Check("scale", scale, E("scale"), 1e-6);
    Check("scaled", scaled, E("scaled"), 1e-5);

    // 2. Masks and patch embedding, from the reference's scaled input.
    var (pv, pm, pg, pt) = (model.Patch(E("scaled")), model.Patch(mask), model.Patch(groups), model.Patch(types));
    var (timeMask, groupMask) = T0Preprocess.AttentionMasks(pg, pt, pm);
    Check("time_mask", timeMask.to(ScalarType.Float32), E("time_mask"), 0);
    Check("group_mask", groupMask.to(ScalarType.Float32), E("group_mask"), 0);
    Check("patch_embedding", model.patch_encoder.forward(pv, pm, pt), E("patch_embedding"), 1e-5);

    // 3. Layers, each from the reference's previous output: time, time, group, ..., last.
    var layers = model.transformer.layers;
    var rope = model.transformer.Rope(pv.shape[1], device);
    var h = E("patch_embedding");
    for (var i = 0; i < layers.Count; i++)
    {
        var refName = $"layer{i}_output";
        h = layers[i].Forward(h, timeMask, groupMask, rope);
        if (!fx.Has(refName)) continue;
        Check(refName, h, E(refName), 1e-4, layers[i].IsGroup ? "(group)" : "(time)");
        h = E(refName);
    }
    Check("transformer_output", model.transformer.Forward(E("patch_embedding"), timeMask, groupMask),
        E("transformer_output"), 2e-4, "(whole stack)");

    // 4. Decoder, quantile head and rescale.
    var decoded = model.decoder.forward(E("transformer_output"));
    Check("decoder_output", decoded, E("decoder_output"), 1e-5);
    var head = T0Model.Monotone(E("decoder_output").view(
        decoded.shape[0], decoded.shape[1], cfg.PatchSize, cfg.NumQuantiles));
    Check("head_output", head, E("head_output"), 1e-5);
    Check("rescaled", T0Preprocess.Rescale(E("head_output"), E("loc"), E("scale"), cfg.PatchSize,
        cfg.ScalerUseArcsinh), E("rescaled"), 1e-5);

    // 5. End to end through the public API, from raw series.
    var ctx = fx.Get("in_ctx");                 // (2, T)
    var fut = fx.Get("in_fut");                 // (2, T + H)
    var (t, horizon) = ((int)ctx.shape[1], (int)(fut.shape[1] - ctx.shape[1]));
    var total = t + horizon;
    var joint = new float[4 * total];
    for (var v = 0; v < 2; v++)
    for (var i = 0; i < t; i++) joint[v * total + i] = ctx[v, i].item<float>();
    for (var v = 0; v < 2; v++)
    for (var i = 0; i < total; i++) joint[(2 + v) * total + i] = fut[v, i].item<float>();

    for (var target = 0; target < 2; target++)
    {
        Check($"joint[{target}]", Grid(forecaster.ForecastJoint(joint, 4, t, horizon,
            [false, false, true, true], target)), E("predict_joint")[target], 1e-4, "(2 targets + 2 known)");
        var plain = joint.AsSpan(0, 2 * total).ToArray();
        Check($"targets_only[{target}]", Grid(forecaster.ForecastJoint(plain, 2, t, horizon,
            targetVariate: target)), E("predict_targets_only")[target], 1e-4);
    }
    var aligned = Enumerable.Range(64, 64).Select(i => ctx[0, i].item<float>()).ToArray();
    Check("aligned", Grid(forecaster.ForecastJoint(aligned, 1, 64, 32)), E("predict_aligned"), 1e-4,
        "(no padding)");

    // 6. The quickstart notebook's examples.
    var m4 = Floats(fx.Get("m4_ctx"));
    Check("notebook m4 H1", Grid(forecaster.ForecastJoint(m4, 1, m4.Length, 48)), E("m4_predict"), 1e-4);

    var ep = Floats(fx.Get("ep_ctx"));
    var cov = fx.Get("ep_cov");                 // (2, 512 + 24)
    var epTruth = Floats(fx.Get("ep_truth"));
    var epPlain = forecaster.ForecastJoint(ep, 1, ep.Length, epTruth.Length);
    Check("notebook ep", Grid(epPlain), E("ep_predict"), 1e-4);

    var width = (int)cov.shape[1];
    var epJoint = new float[3 * width];
    ep.CopyTo(epJoint, 0);
    Floats(cov[0]).CopyTo(epJoint, width);
    Floats(cov[1]).CopyTo(epJoint, 2 * width);
    var epCov = forecaster.ForecastJoint(epJoint, 3, ep.Length, epTruth.Length, [false, true, true]);
    Check("notebook ep + cov", Grid(epCov), E("ep_predict_cov"), 1e-4);

    var mid = Array.IndexOf(cfg.Quantiles, 0.5);
    Console.WriteLine($"  electricity price |median-truth|: {MeanAbs(epPlain, epTruth, mid):F4} without covariates, "
                      + $"{MeanAbs(epCov, epTruth, mid):F4} with");
    return pass;
}

static float[] Floats(Tensor x) => x.contiguous().data<float>().ToArray();

static Tensor Grid(double[,] q) => tensor(q.Cast<double>().Select(d => (float)d).ToArray(),
    [q.GetLength(0), q.GetLength(1)]);

static double MeanAbs(double[,] q, float[] truth, int col)
    => Enumerable.Range(0, truth.Length).Average(h => Math.Abs(q[h, col] - truth[h]));

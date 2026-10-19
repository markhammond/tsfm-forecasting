using System.Text.Json;

using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Tsfm.Forecasting.T0;

/// <summary>
/// Checkpoint hyper-parameters, read from the published config.json.
///
/// <para>Absent fields take t0-alpha's values, as the reference does: alpha's config
/// predates them. t0-beta states <c>scaler_eps</c> 0.01 under <c>std_clamp</c>; run
/// under alpha's defaults (0.1, <c>variance_offset</c>) it loads and forecasts without
/// complaint, just worse — so these are read, never assumed.</para>
/// </summary>
public sealed record T0Config(
    long EmbedDim, int NumLayers, long NumHeads, long MlpHiddenDim, int PatchSize, int GroupEveryN,
    double[] Quantiles, bool ScalerUseArcsinh, double ScalerEps, bool ScalerStdClamp,
    bool XposHalvesLayout, bool ScalerZeroNoValue)
{
    public int NumQuantiles => Quantiles.Length;

    public static T0Config Load(string path) => Parse(File.ReadAllText(path));

    public static T0Config Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var r = doc.RootElement;
        string? Text(string name) => r.TryGetProperty(name, out var e) && e.ValueKind == JsonValueKind.String
            ? e.GetString() : null;
        bool Flag(string name, bool fallback) => r.TryGetProperty(name, out var e) ? e.GetBoolean() : fallback;

        var epsMode = Text("scaler_eps_mode") ?? "variance_offset";
        var layout = Text("xpos_scale_layout") ?? "interleaved";
        if (epsMode is not ("variance_offset" or "std_clamp"))
            throw new InvalidDataException($"unknown scaler_eps_mode '{epsMode}'");
        if (layout is not ("interleaved" or "halves"))
            throw new InvalidDataException($"unknown xpos_scale_layout '{layout}'");

        return new T0Config(
            r.GetProperty("embed_dim").GetInt64(),
            r.GetProperty("num_layers").GetInt32(),
            r.GetProperty("num_heads").GetInt64(),
            r.GetProperty("mlp_hidden_dim").GetInt64(),
            r.GetProperty("patch_size").GetInt32(),
            r.GetProperty("group_every_n").GetInt32(),
            [.. r.GetProperty("quantile_levels").EnumerateArray().Select(q => q.GetDouble())],
            Flag("scaler_use_arcsinh", true),
            r.TryGetProperty("scaler_eps", out var eps) ? eps.GetDouble() : 0.1,
            epsMode == "std_clamp",
            layout == "halves",
            Flag("scaler_zero_no_value", true));
    }
}

/// <summary>
/// t0: patch encoder, alternating time/group transformer, and a residual decoder whose
/// output is made monotone across quantiles.
///
/// <para>Next-patch prediction: the output at patch <c>i</c> forecasts patch <c>i + 1</c>.
/// Every patch of a known horizon is decoded in the same pass, so a predictive interval over
/// up to <see cref="T0Forecaster.MaxHorizon"/> steps costs one forward.</para>
/// </summary>
public sealed class T0Model : Module
{
    private readonly T0Config _cfg;

    public PatchEncoder patch_encoder;
    public Transformer transformer;
    public ResidualBlock decoder;

    public T0Model(T0Config cfg) : base(nameof(T0Model))
    {
        _cfg = cfg;
        patch_encoder = new PatchEncoder(cfg.EmbedDim, cfg.PatchSize);
        transformer = new Transformer(cfg);
        decoder = new ResidualBlock(cfg.EmbedDim, cfg.EmbedDim, cfg.PatchSize * cfg.NumQuantiles);
        RegisterComponents();
    }

    public T0Config Config => _cfg;

    /// <summary>Split a per-step array <c>(rows, width)</c> into patches.</summary>
    public Tensor Patch(Tensor x) => x.view(x.shape[0], x.shape[1] / _cfg.PatchSize, _cfg.PatchSize);

    /// <summary>
    /// Per-patch quantiles in scaled space.
    /// </summary>
    /// <param name="scaled">Scaled values (rows, width), width a multiple of the patch.</param>
    /// <param name="mask"><see cref="MaskType"/> per step.</param>
    /// <param name="groupIds">Series id per step; rows sharing one attend to each other.
    /// -1 on padding.</param>
    /// <param name="variateType"><see cref="VariateType"/> per step, -1 on padding.</param>
    /// <returns>(rows, patches, patch, quantiles), non-decreasing along the last axis.</returns>
    public Tensor Backbone(Tensor scaled, Tensor mask, Tensor groupIds, Tensor variateType)
    {
        using var scope = NewDisposeScope();
        var (values, m, g, t) = (Patch(scaled), Patch(mask), Patch(groupIds), Patch(variateType));
        var embedded = patch_encoder.forward(values, m, t);
        var (timeMask, groupMask) = T0Preprocess.AttentionMasks(g, t, m);
        var h = transformer.Forward(embedded, timeMask, groupMask);
        var decoded = decoder.forward(h);
        decoded = decoded.view(decoded.shape[0], decoded.shape[1], _cfg.PatchSize, _cfg.NumQuantiles);
        return Monotone(decoded).MoveToOuterDisposeScope();
    }

    /// <summary>The quantile head: the first column free, each later one the previous plus
    /// a softplus, so quantiles cannot cross.</summary>
    public static Tensor Monotone(Tensor x)
    {
        using var scope = NewDisposeScope();
        var q = x.shape[^1];
        var first = x.narrow(-1, 0, 1);
        if (q == 1) return first.MoveToOuterDisposeScope();
        var rest = first + functional.softplus(x.narrow(-1, 1, q - 1)).cumsum(-1);
        return cat([first, rest], -1).MoveToOuterDisposeScope();
    }

    /// <summary>Bind published weights by their PyTorch names. Every parameter must be
    /// matched: an unbound tensor keeps its random init and still forecasts plausibly.</summary>
    public static T0Model Load(string checkpointDir, Device device)
    {
        var cfg = T0Config.Load(Path.Combine(checkpointDir, "config.json"));
        using var stream = File.OpenRead(Path.Combine(checkpointDir, "model.safetensors"));
        return Load(cfg, stream, device, checkpointDir);
    }

    /// <inheritdoc cref="Load(string, Device)"/>
    public static T0Model Load(T0Config cfg, Stream weights, Device device, string label = "<stream>")
    {
        var net = new T0Model(cfg);
        using var st = Safetensors.Load(weights, label);

        // The head's levels ride in the checkpoint as a buffer. A config that disagrees with
        // them would label every column wrongly while computing it correctly.
        if (st.Has("head.quantile_levels"))
        {
            using var levels = st.Get("head.quantile_levels");
            var stored = levels.data<float>().ToArray();
            if (stored.Length != cfg.NumQuantiles
                || stored.Where((l, i) => Math.Abs(l - cfg.Quantiles[i]) > 1e-6).Any())
                throw new InvalidDataException(
                    $"{label}: config quantiles [{string.Join(", ", cfg.Quantiles)}] disagree with "
                    + $"the checkpoint's [{string.Join(", ", stored)}]");
        }

        var bound = 0;
        var missing = new List<string>();
        using (no_grad())
        {
            foreach (var (name, param) in net.named_parameters())
            {
                if (!st.Has(name)) { missing.Add(name); continue; }
                using var t = st.Get(name);
                if (!t.shape.SequenceEqual(param.shape))
                    throw new InvalidDataException(
                        $"{name}: checkpoint {string.Join('x', t.shape)} vs model {string.Join('x', param.shape)}");
                param.copy_(t);
                bound++;
            }
        }
        if (missing.Count > 0)
            throw new InvalidDataException(
                $"{missing.Count} parameters absent from {label}, e.g. {string.Join(", ", missing.Take(5))}");

        Console.Error.WriteLine($"bound {bound} parameters from {label}");
        net.to(device);
        net.eval();
        return net;
    }
}

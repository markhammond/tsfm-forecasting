using System.Text.Json;

using TorchSharp;

using static TorchSharp.torch;

namespace Tsfm.Forecasting.Timestral;

/// <summary>
/// Checkpoint hyper-parameters, read from the published config.json. Strict, as the
/// reference is: an unknown key may change the forward pass, so it is refused rather than
/// ignored.
/// </summary>
public sealed record TimestralConfig(
    int PatchSize, long DModel, long NumHeads, long MlpHiddenDim, string[] BlockSchedule, double[] Quantiles,
    int BlockSize, int ContextLength, double RopeTheta, double RopePartialFactor, double NormEps,
    bool AttnBias, bool MlpBias, bool TimeAttnGate, bool VariateAttnGate, double MinimumScale, bool UseAsinh,
    bool CovariateEnabled, bool SplitContextHorizonAttn, bool AttendMissingPatches, double QuantileWidthFactor)
{
    public long HeadDim => DModel / NumHeads;
    public long RopeDim => (long)(HeadDim * RopePartialFactor);
    public int NumQuantiles => Quantiles.Length;
    public int MedianIndex => Array.IndexOf(Quantiles, 0.5);

    /// <summary>Most steps decoded in one forward pass.</summary>
    public int MaxHorizon => BlockSize * PatchSize;

    public static TimestralConfig Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var r = doc.RootElement;
        string[] known =
        [
            "patch_size", "d_model", "num_heads", "mlp_hidden_dim", "block_schedule", "quantile_knots", "block_size",
            "context_length", "rope_theta", "rope_partial_factor", "norm_eps", "attn_bias", "mlp_bias",
            "time_attn_gate", "variate_attn_gate", "minimum_scale", "use_asinh", "stabilize_with_global",
            "covariate_enabled", "split_context_horizon_attn", "attend_missing_patches", "quantile_width_factor",
        ];
        var unknown = r.EnumerateObject().Select(p => p.Name).Except(known).ToArray();
        if (unknown.Length > 0)
            throw new InvalidDataException($"unknown config keys: {string.Join(", ", unknown)}");

        double D(string k, double d) => r.TryGetProperty(k, out var e) ? e.GetDouble() : d;
        bool B(string k, bool d) => r.TryGetProperty(k, out var e) ? e.GetBoolean() : d;
        bool Gate(string k)
        {
            var g = r.TryGetProperty(k, out var e) ? e.GetString() : "sdpa_sigmoid";
            return g switch
            {
                "sdpa_sigmoid" => true,
                "none" => false,
                _ => throw new InvalidDataException($"unknown attention gate '{g}'"),
            };
        }

        // The advancing refine frame the forecaster decodes in assumes purely causal
        // statistics; the global stabiliser would change every patch's frame.
        if (B("stabilize_with_global", false))
            throw new InvalidDataException("stabilize_with_global is not supported by the advancing refine frame");

        var cfg = new TimestralConfig(
            r.GetProperty("patch_size").GetInt32(),
            r.GetProperty("d_model").GetInt64(),
            r.GetProperty("num_heads").GetInt64(),
            r.GetProperty("mlp_hidden_dim").GetInt64(),
            [.. r.GetProperty("block_schedule").EnumerateArray().Select(e => e.GetString()!)],
            [.. r.GetProperty("quantile_knots").EnumerateArray().Select(e => e.GetDouble())],
            r.GetProperty("block_size").GetInt32(),
            r.GetProperty("context_length").GetInt32(),
            D("rope_theta", 10000.0), D("rope_partial_factor", 0.5), D("norm_eps", 1e-6),
            B("attn_bias", false), B("mlp_bias", false), Gate("time_attn_gate"), Gate("variate_attn_gate"),
            D("minimum_scale", 1e-6), B("use_asinh", true), B("covariate_enabled", true),
            B("split_context_horizon_attn", true), B("attend_missing_patches", true), D("quantile_width_factor", 2.0));

        if (cfg.MedianIndex < 0) throw new InvalidDataException("quantile_knots must contain 0.5");
        if (!cfg.AttendMissingPatches)
            throw new InvalidDataException("attend_missing_patches=false is not ported");
        return cfg;
    }
}

/// <summary>
/// Timestral: a causal patch transformer trained with Decoupled Horizon Forecasting. Context
/// patches are encoded causally; the horizon is read by learned forecast-query tokens that
/// attend to the encoded context, and a monotone head emits 21 quantiles per step.
///
/// <para>Each row is a target, a past-only covariate or a known-future covariate. With
/// known-future rows the encoder also spans the horizon, so their future values are keys;
/// targets' horizon slots stay blank.</para>
/// </summary>
public sealed class TimestralModel : nn.Module
{
    private readonly TimestralConfig _c;

    public PatchRepresentation representation;
    public Transformer transformer;
    public QuantileHead head;

    public TimestralModel(TimestralConfig c) : base(nameof(TimestralModel))
    {
        _c = c;
        representation = new PatchRepresentation(c.PatchSize, c.DModel);
        transformer = new Transformer(c);
        head = new QuantileHead(c);
        RegisterComponents();
    }

    public TimestralConfig Config => _c;

    /// <summary>
    /// Forecast one block of <paramref name="horizonPatches"/> after packed rows.
    /// </summary>
    /// <param name="values">(rows, W * patch): the context, then — when any row is a
    /// known-future covariate — the horizon, carrying those rows' future values.</param>
    /// <param name="observed">Observation mask, same shape.</param>
    /// <param name="groups">(rows) series id; rows of a series attend each other.</param>
    /// <param name="roles">(rows) <see cref="Role"/>.</param>
    /// <param name="contextPatches">Context patches, L. W is L, or L + h with known rows.</param>
    /// <param name="horizonPatches">Patches to forecast, h.</param>
    /// <returns>Quantiles (rows, h * patch, Q) in data units, in the frame of the last
    /// context patch.</returns>
    public Tensor PredictBlock(Tensor values, Tensor observed, Tensor groups, Tensor roles,
        int contextPatches, int horizonPatches)
    {
        using var scope = NewDisposeScope();
        var (r, p) = (values.shape[0], _c.PatchSize);
        var w = (int)(values.shape[1] / p);

        var (scaled, loc, scale) = TimestralPreprocess.Scale(values, observed, roles, _c);
        var xEnc = representation.Embed(scaled, observed);
        var xDec = representation.ForecastTokens(r, horizonPatches);

        var perPatch = observed.reshape(r, w, p).any(-1);
        var keyValid = TimestralPreprocess.KeyValid(perPatch);
        var timeMask = TimestralPreprocess.TimeMask(keyValid, roles.eq(Role.KnownCovariate), contextPatches, horizonPatches);
        var variateMask = TimestralPreprocess.VariateMask(groups, keyValid, roles, horizonPatches);
        var first = TimestralPreprocess.FirstObserved(perPatch, groups);
        var positions = TimestralPreprocess.Positions(first, w, contextPatches, horizonPatches);

        var h = transformer.Forward(cat([xEnc, xDec], 1), timeMask, variateMask, positions, w);
        var preds = head.Forward(h.narrow(1, w, horizonPatches));

        var anchor = (contextPatches - 1) * p;
        var anchorLoc = loc.narrow(1, anchor, 1).unsqueeze(-1);
        var anchorScale = scale.narrow(1, anchor, 1).unsqueeze(-1);
        return TimestralPreprocess.Denormalize(preds, anchorLoc, anchorScale, _c.UseAsinh).MoveToOuterDisposeScope();
    }

    /// <summary>Bind published weights by their PyTorch names. Every parameter must be
    /// matched, and every checkpoint tensor used: an unbound one keeps its random init and
    /// still forecasts plausibly.</summary>
    public static TimestralModel Load(IModelWeights weights, Device device)
    {
        var cfg = TimestralConfig.Parse(weights.ModelConfigJson);
        var net = new TimestralModel(cfg);
        using var stream = weights.OpenModel();
        using var st = Safetensors.Load(stream, weights.Name);

        var named = net.named_parameters().ToDictionary(p => p.name, p => p.parameter);
        var missing = named.Keys.Where(k => !st.Has(k)).ToArray();
        var extra = st.Tensors.Keys.Where(k => !named.ContainsKey(k)).ToArray();
        if (missing.Length > 0 || extra.Length > 0)
            throw new InvalidDataException(
                $"{weights.Name}: {missing.Length} parameters absent (e.g. {string.Join(", ", missing.Take(3))}), "
                + $"{extra.Length} unexpected (e.g. {string.Join(", ", extra.Take(3))})");

        using (no_grad())
        {
            foreach (var (name, param) in named)
            {
                using var t = st.Get(name);
                if (!t.shape.SequenceEqual(param.shape))
                    throw new InvalidDataException(
                        $"{name}: checkpoint {string.Join('x', t.shape)} vs model {string.Join('x', param.shape)}");
                param.copy_(t);
            }
        }

        Console.Error.WriteLine($"bound {named.Count} parameters from {weights.Name}");
        net.to(device);
        net.eval();
        return net;
    }
}

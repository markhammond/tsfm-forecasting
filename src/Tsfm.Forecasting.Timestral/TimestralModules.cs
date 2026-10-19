using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Tsfm.Forecasting.Timestral;

/// <summary>RMS norm, weight named <c>scale</c>.</summary>
public sealed class RmsNorm(long dims, double eps) : Module<Tensor, Tensor>(nameof(RmsNorm))
{
    public Parameter scale = new(ones(dims));

    public override Tensor forward(Tensor x)
    {
        using var s = NewDisposeScope();
        return (x * (x.pow(2).mean([-1L], keepdim: true) + eps).rsqrt() * scale).MoveToOuterDisposeScope();
    }
}

/// <summary><c>output(silu(hidden(x))) + residual(x)</c> — SiLU, where t0's uses ReLU.</summary>
public sealed class ResidualBlock : Module<Tensor, Tensor>
{
    public Linear hidden_layer, output_layer, residual_layer;

    public ResidualBlock(long input, long hidden, long output, bool bias) : base(nameof(ResidualBlock))
    {
        hidden_layer = Linear(input, hidden, hasBias: bias);
        output_layer = Linear(hidden, output, hasBias: bias);
        residual_layer = Linear(input, output, hasBias: bias);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var s = NewDisposeScope();
        return (output_layer.forward(functional.silu(hidden_layer.forward(x))) + residual_layer.forward(x))
            .MoveToOuterDisposeScope();
    }
}

/// <summary>SwiGLU with one fused input projection, computing <c>gate * silu(up)</c>: the
/// SiLU falls on the SECOND half, the reverse of t0's.</summary>
public sealed class FusedSwiGlu : Module<Tensor, Tensor>
{
    public Linear w12, w3;

    public FusedSwiGlu(long embed, long hidden, bool bias) : base(nameof(FusedSwiGlu))
    {
        w12 = Linear(embed, 2 * hidden, hasBias: bias);
        w3 = Linear(hidden, embed, hasBias: bias);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var s = NewDisposeScope();
        var halves = w12.forward(x).chunk(2, -1);
        return w3.forward(halves[0] * functional.silu(halves[1])).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// Partial rotary embedding in the split-half layout: dimension <c>i</c> pairs with
/// <c>i + headDim/2</c>, and only the first <c>rotaryDim/2</c> pairs turn — the rest see
/// cos 1, sin 0. Positions are integers per row, anchored at the series' first observed
/// patch, so left padding does not shift them.
/// </summary>
public sealed class PartialRope(long headDim, long rotaryDim, double theta)
{
    /// <summary>Cos and sin, each <c>(rows, seq, headDim)</c>, for integer positions
    /// <c>(rows, seq)</c>.</summary>
    public (Tensor Cos, Tensor Sin) Tables(Tensor positions)
    {
        using var s = NewDisposeScope();
        var idx = arange(0, rotaryDim, 2, ScalarType.Float32, positions.device) / rotaryDim;
        var rotated = 1.0 / full_like(idx, theta).pow(idx);
        var inv = cat([rotated, zeros(headDim / 2 - rotated.shape[0], ScalarType.Float32, positions.device)]);
        var freqs = positions.to(ScalarType.Float32).unsqueeze(-1) * inv;
        var emb = cat([freqs, freqs], -1);
        return (emb.cos().MoveToOuterDisposeScope(), emb.sin().MoveToOuterDisposeScope());
    }

    public static Tensor Apply(Tensor x, Tensor cos, Tensor sin)
    {
        using var s = NewDisposeScope();
        var half = x.shape[^1] / 2;
        var rotated = cat([-x.narrow(-1, half, half), x.narrow(-1, 0, half)], -1);
        return (x * cos + rotated * sin).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// Gated multi-head attention over time (rows attend along patches, with RoPE) or across
/// variates (patches attend along rows, unordered). The output is multiplied by
/// <c>sigmoid(_gate(x))</c> before the output projection. Unbiased throughout.
/// </summary>
public sealed class Attention : Module
{
    private readonly long _heads, _headDim, _embed;

    public bool IsTime { get; }

    public Linear wQKV, wO;
    public Linear? _gate;

    public Attention(long embed, long heads, bool isTime, bool bias, bool gated) : base(nameof(Attention))
    {
        (_embed, _heads, _headDim, IsTime) = (embed, heads, embed / heads, isTime);
        wQKV = Linear(embed, 3 * embed, hasBias: bias);
        wO = Linear(embed, embed, hasBias: bias);
        if (gated) _gate = Linear(embed, embed, hasBias: bias);
        RegisterComponents();
    }

    /// <param name="x">(rows, patches, embed)</param>
    /// <param name="mask">Time: (rows, 1, S, S). Variate: (S, 1, rows, rows). True attends.</param>
    /// <param name="cos">RoPE table (rows, S, headDim); time only.</param>
    /// <param name="sin">RoPE table; time only.</param>
    /// <param name="encLen">Encoder tokens leading the time sequence. Their queries read
    /// encoder keys only; the rest read everything — the quadrant they skip is masked anyway.</param>
    public Tensor Forward(Tensor x, Tensor mask, Tensor? cos, Tensor? sin, long encLen)
    {
        using var s = NewDisposeScope();
        var (rows, patches) = (x.shape[0], x.shape[1]);
        var inputs = IsTime ? x : x.permute(1, 0, 2);
        var (b, n) = (inputs.shape[0], inputs.shape[1]);

        // The fused projection is laid out (qkv, head_dim, heads): head-minor, not head-major.
        var qkv = wQKV.forward(inputs.contiguous()).reshape(b, n, 3, _headDim, _heads).permute(2, 0, 1, 4, 3);
        var (q, k, v) = (qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous());

        Tensor o;
        if (IsTime)
        {
            var (c, sn) = (cos!.unsqueeze(2), sin!.unsqueeze(2));
            (q, k) = (PartialRope.Apply(q, c, sn), PartialRope.Apply(k, c, sn));
            var dec = n - encLen;
            var enc = Sdpa(q.narrow(1, 0, encLen), k.narrow(1, 0, encLen), v.narrow(1, 0, encLen),
                mask.narrow(2, 0, encLen).narrow(3, 0, encLen));
            o = dec == 0 ? enc : cat([enc, Sdpa(q.narrow(1, encLen, dec), k, v, mask.narrow(2, encLen, dec))], 1);
        }
        else
        {
            o = Sdpa(q, k, v, mask);
        }

        if (_gate is not null) o = o * _gate.forward(inputs).sigmoid();
        o = IsTime ? o.reshape(rows, patches, _embed) : o.reshape(patches, rows, _embed).permute(1, 0, 2);
        return wO.forward(o).MoveToOuterDisposeScope();
    }

    /// <summary>SDPA on <c>(batch, seq, heads, headDim)</c>; returns <c>(batch, seq, heads * headDim)</c>.</summary>
    private static Tensor Sdpa(Tensor q, Tensor k, Tensor v, Tensor mask)
    {
        var (b, n) = (q.shape[0], q.shape[1]);
        var o = functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask: mask);
        return o.transpose(1, 2).contiguous().view(b, n, -1);
    }
}

/// <summary>One layer: its attentions in schedule order, each a pre-norm residual, then
/// one pre-norm SwiGLU.</summary>
public sealed class TransformerBlock : Module
{
    public ModuleList<RmsNorm> attn_norms;
    public ModuleList<Attention> attns;
    public RmsNorm mlp_norm;
    public FusedSwiGlu mlp;

    public TransformerBlock(string kind, TimestralConfig c) : base(nameof(TransformerBlock))
    {
        bool[] axes = kind switch
        {
            "time" => [true],
            "variate" => [false],
            "time_variate" => [true, false],
            "variate_time" => [false, true],
            _ => throw new InvalidDataException($"unknown layer kind '{kind}'"),
        };
        attn_norms = new ModuleList<RmsNorm>([.. axes.Select(_ => new RmsNorm(c.DModel, c.NormEps))]);
        attns = new ModuleList<Attention>([.. axes.Select(t => new Attention(c.DModel, c.NumHeads, t, c.AttnBias,
            t ? c.TimeAttnGate : c.VariateAttnGate))]);
        mlp_norm = new RmsNorm(c.DModel, c.NormEps);
        mlp = new FusedSwiGlu(c.DModel, c.MlpHiddenDim, c.MlpBias);
        RegisterComponents();
    }

    public Tensor Forward(Tensor x, Tensor timeMask, Tensor variateMask, Tensor cos, Tensor sin, long encLen)
    {
        using var s = NewDisposeScope();
        for (var i = 0; i < attns.Count; i++)
        {
            var a = attns[i];
            x = x + a.Forward(attn_norms[i].forward(x), a.IsTime ? timeMask : variateMask, cos, sin, encLen);
        }
        return (x + mlp.forward(mlp_norm.forward(x))).MoveToOuterDisposeScope();
    }
}

/// <summary>The layer stack and its final norm.</summary>
public sealed class Transformer : Module
{
    private readonly PartialRope _rope;

    public ModuleList<TransformerBlock> layers;
    public RmsNorm out_norm;

    public Transformer(TimestralConfig c) : base(nameof(Transformer))
    {
        _rope = new PartialRope(c.HeadDim, c.RopeDim, c.RopeTheta);
        layers = new ModuleList<TransformerBlock>([.. c.BlockSchedule.Select(k => new TransformerBlock(k, c))]);
        out_norm = new RmsNorm(c.DModel, c.NormEps);
        RegisterComponents();
    }

    /// <param name="x"><c>[Enc ; Dec]</c> tokens (rows, S, embed).</param>
    /// <param name="timeMask">(rows, 1, S, S).</param>
    /// <param name="variateMask">(S, 1, rows, rows).</param>
    /// <param name="positions">Integer time positions (rows, S).</param>
    /// <param name="encLen">Encoder tokens leading the sequence.</param>
    public Tensor Forward(Tensor x, Tensor timeMask, Tensor variateMask, Tensor positions, long encLen)
    {
        using var s = NewDisposeScope();
        var (cos, sin) = _rope.Tables(positions);
        foreach (var layer in layers) x = layer.Forward(x, timeMask, variateMask, cos, sin, encLen);
        return out_norm.forward(x).MoveToOuterDisposeScope();
    }
}

/// <summary>Patch embedding from <c>[values ‖ missing]</c>, and the learned forecast-query
/// token every horizon patch starts from.</summary>
public sealed class PatchRepresentation : Module
{
    private readonly int _patch;

    public ResidualBlock patch_proj;
    public Parameter forecast_query;

    public PatchRepresentation(int patch, long dModel) : base(nameof(PatchRepresentation))
    {
        _patch = patch;
        patch_proj = new ResidualBlock(2 * patch, 4 * dModel, dModel, bias: true);
        forecast_query = new Parameter(zeros(dModel));
        RegisterComponents();
    }

    /// <summary><c>(rows, T)</c> scaled values and observation mask to <c>(rows, T / patch, dModel)</c>.</summary>
    public Tensor Embed(Tensor scaled, Tensor observed)
    {
        using var s = NewDisposeScope();
        var (r, n) = (scaled.shape[0], scaled.shape[1] / _patch);
        var missing = (~observed).to(scaled.dtype).reshape(r, n, _patch);
        return patch_proj.forward(cat([scaled.reshape(r, n, _patch), missing], -1)).MoveToOuterDisposeScope();
    }

    public Tensor ForecastTokens(long rows, long patches) => forecast_query.view(1, 1, -1).expand(rows, patches, -1);
}

/// <summary>Decoder tokens to <c>patch</c> monotone quantile vectors each, in scaled space.</summary>
public sealed class QuantileHead : Module
{
    private readonly TimestralConfig _c;

    public ResidualBlock projection;

    public QuantileHead(TimestralConfig c) : base(nameof(QuantileHead))
    {
        _c = c;
        projection = new ResidualBlock(c.DModel, 4 * c.DModel, c.PatchSize * c.NumQuantiles, bias: false);
        RegisterComponents();
    }

    /// <summary><c>(rows, N, dModel)</c> to <c>(rows, N * patch, Q)</c>.</summary>
    public Tensor Forward(Tensor x)
    {
        using var s = NewDisposeScope();
        var raw = projection.forward(x);
        var (r, n) = (raw.shape[0], raw.shape[1]);
        raw = raw.reshape(r, n, _c.PatchSize, _c.NumQuantiles);
        return Monotone(raw, _c.Quantiles, _c.QuantileWidthFactor)
            .reshape(r, n * _c.PatchSize, _c.NumQuantiles).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Non-crossing quantiles: the median is free, and the rest fan out from it by
    /// <c>cumsum(width * dp * softplus(raw))</c>, <c>dp</c> the probability gap to the next
    /// level inward, so unequal knot spacing is respected.
    /// </summary>
    public static Tensor Monotone(Tensor raw, double[] levels, double width)
    {
        using var s = NewDisposeScope();
        var m = Array.IndexOf(levels, 0.5);
        var gaps = new float[levels.Length];
        for (var i = m + 1; i < levels.Length; i++) gaps[i] = (float)(levels[i] - levels[i - 1]);
        for (var i = 0; i < m; i++) gaps[i] = (float)(levels[i + 1] - levels[i]);

        var inc = width * tensor(gaps).to(raw.device) * functional.softplus(raw);
        var median = raw.narrow(-1, m, 1);
        var up = inc.narrow(-1, m + 1, levels.Length - m - 1).cumsum(-1);
        var down = inc.narrow(-1, 0, m).flip(-1).cumsum(-1).flip(-1);
        var big = float.MaxValue;
        return cat([median - down, median, median + up], -1).clamp(-big, big).MoveToOuterDisposeScope();
    }
}

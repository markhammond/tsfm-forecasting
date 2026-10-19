using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Tsfm.Forecasting.T0;

/// <summary>RMS norm, weight named <c>scale</c>, eps 1e-8 — not TimesFM's 1e-6.</summary>
public sealed class RmsNorm(long dims, double eps = 1e-8) : Module<Tensor, Tensor>(nameof(RmsNorm))
{
    public Parameter scale = new(ones(dims));

    public override Tensor forward(Tensor x)
    {
        using var s = NewDisposeScope();
        return (x / (x * x).mean([-1L], keepdim: true).add(eps).sqrt() * scale).MoveToOuterDisposeScope();
    }
}

/// <summary>Two biased linear layers with a ReLU between.</summary>
public sealed class Mlp : Module<Tensor, Tensor>
{
    public Linear hidden_layer, output_layer;

    public Mlp(long input, long hidden, long output) : base(nameof(Mlp))
    {
        hidden_layer = Linear(input, hidden);
        output_layer = Linear(hidden, output);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var s = NewDisposeScope();
        return output_layer.forward(functional.relu(hidden_layer.forward(x))).MoveToOuterDisposeScope();
    }
}

/// <summary><c>mlp(x) + residual_layer(x)</c>, every projection biased.</summary>
public sealed class ResidualBlock : Module<Tensor, Tensor>
{
    public Mlp mlp;
    public Linear residual_layer;

    public ResidualBlock(long input, long hidden, long output) : base(nameof(ResidualBlock))
    {
        mlp = new Mlp(input, hidden, output);
        residual_layer = Linear(input, output);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var s = NewDisposeScope();
        return (mlp.forward(x) + residual_layer.forward(x)).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// Embeds each patch as <c>[values ‖ within-patch time ‖ validity]</c>, then adds a learned
/// embedding of the row's role (target, historical, known-future).
/// </summary>
public sealed class PatchEncoder : Module<Tensor, Tensor, Tensor, Tensor>
{
    private readonly int _patch;

    public ResidualBlock projection;
    public Embedding type_embeddings;

    public PatchEncoder(long embedDim, int patchSize) : base(nameof(PatchEncoder))
    {
        _patch = patchSize;
        projection = new ResidualBlock(3 * patchSize, embedDim, embedDim);
        type_embeddings = Embedding(3, embedDim);
        RegisterComponents();
    }

    /// <param name="values">Scaled values (rows, patches, patch).</param>
    /// <param name="mask">Patched <see cref="MaskType"/>.</param>
    /// <param name="variateType">Patched <see cref="VariateType"/>, -1 on padding.</param>
    public override Tensor forward(Tensor values, Tensor mask, Tensor variateType)
    {
        using var s = NewDisposeScope();
        var (v, p) = (values.shape[0], values.shape[1]);
        var time = (arange(_patch, values.dtype, values.device) / _patch).expand(v, p, _patch);
        var validity = mask.eq(MaskType.Valid).to(values.dtype);
        var embedded = projection.forward(cat([values, time, validity], -1));
        // Left padding must not retype a covariate as a target: the role comes from the first
        // real step, and an all-padding patch (-1) is clamped to 0 and masked downstream.
        var role = T0Preprocess.ReducePatchMetadata(variateType, mask).clamp_min(0);
        return (embedded + type_embeddings.forward(role)).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// Rotary embedding over patches with xPos scaling: queries are scaled up and keys down by
/// a per-frequency factor that decays with distance from the window's centre.
///
/// <para>Interleaved, not split-half: pairs are adjacent dimensions <c>(2i, 2i+1)</c>, as
/// rotary-embedding-torch forms them. TimesFM pairs the halves; reusing its RoPE here would
/// load and run without complaint.</para>
/// </summary>
public sealed class XposRotary(long headDim, bool halvesLayout, double theta = 10000.0, double scaleBase = 512.0)
{
    /// <summary>Cos, sin and xPos scale for <paramref name="length"/> positions, each
    /// <c>(length, headDim)</c>. Computed per call: the window length varies.</summary>
    public (Tensor Cos, Tensor Sin, Tensor Scale) Tables(long length, Device device)
    {
        using var s = NewDisposeScope();
        var idx = arange(0, headDim, 2, ScalarType.Int64, device);
        var exponent = idx.to(ScalarType.Float32) / headDim;
        var inv = 1.0 / full_like(exponent, theta).pow(exponent);
        var pos = arange(length, ScalarType.Float32, device);
        var angles = (pos.unsqueeze(-1) * inv).repeat_interleave(2, -1);

        var basis = (idx + 0.4 * headDim) / (1.4 * headDim);
        var power = (pos - Math.Floor((length - 1) / 2.0)) / scaleBase;
        var half = basis.pow(power.unsqueeze(-1));
        var scale = halvesLayout ? cat([half, half], -1) : half.repeat_interleave(2, -1);

        return (angles.cos().MoveToOuterDisposeScope(), angles.sin().MoveToOuterDisposeScope(),
                scale.MoveToOuterDisposeScope());
    }

    /// <summary>Rotate <c>(…, seq, headDim)</c> by the tables; <paramref name="scale"/> is the
    /// xPos factor, inverted for keys.</summary>
    public static Tensor Apply(Tensor t, Tensor cos, Tensor sin, Tensor scale)
    {
        using var s = NewDisposeScope();
        var pairs = t.reshape([.. t.shape[..^1], t.shape[^1] / 2, 2]);
        var rotated = stack([-pairs.select(-1, 1), pairs.select(-1, 0)], -1).flatten(-2);   // (-x2, x1)
        return (t * cos * scale + rotated * sin * scale).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// Multi-head self-attention over the middle axis of <c>(batch, seq, embed)</c>, with a
/// fused biased QKV projection and per-head RMS norm on queries and keys. With rotary
/// tables supplied, they are applied AFTER the norm — TimesFM applies them before.
/// </summary>
public sealed class SelfAttention : Module
{
    private readonly long _heads, _headDim;

    public Linear wQKV, wO;
    public RmsNorm q_norm, k_norm;

    public SelfAttention(long embedDim, long heads) : base(nameof(SelfAttention))
    {
        (_heads, _headDim) = (heads, embedDim / heads);
        wQKV = Linear(embedDim, 3 * embedDim);
        wO = Linear(embedDim, embedDim);
        q_norm = new RmsNorm(_headDim);
        k_norm = new RmsNorm(_headDim);
        RegisterComponents();
    }

    /// <param name="x">(batch, seq, embed)</param>
    /// <param name="mask">(batch, 1, seq, seq), true where attention is allowed.</param>
    /// <param name="rope">Rotary tables, or null for unordered (variate) attention.</param>
    public Tensor Forward(Tensor x, Tensor mask, (Tensor Cos, Tensor Sin, Tensor Scale)? rope)
    {
        using var s = NewDisposeScope();
        var (b, n) = (x.shape[0], x.shape[1]);

        // "(three h d)" on the projection: three-major, then heads.
        var qkv = wQKV.forward(x).view(b, n, 3, _heads, _headDim).permute(2, 0, 3, 1, 4);
        var q = q_norm.forward(qkv[0]);
        var k = k_norm.forward(qkv[1]);
        var v = qkv[2];

        if (rope is var (cos, sin, scale))
        {
            q = XposRotary.Apply(q, cos, sin, scale);
            k = XposRotary.Apply(k, cos, sin, scale.pow(-1));   // keys take the inverse
        }

        var o = functional.scaled_dot_product_attention(q, k, v, attn_mask: mask);
        // A query with no permitted key (a padding patch) must yield zero, as the reference's
        // safe softmax does, not NaN: as a value it would poison every row that reads it.
        o = where(mask.any(-1, keepdim: true), o, zeros_like(o));

        o = o.transpose(1, 2).reshape(b, n, _heads * _headDim);
        return wO.forward(o).MoveToOuterDisposeScope();
    }
}

/// <summary>Pre-norm attention with a residual. Group blocks attend across rows at each
/// patch, by swapping the row and patch axes around the attention.</summary>
public sealed class AttentionBlock : Module
{
    private readonly bool _group;

    public RmsNorm norm;
    public SelfAttention attention;

    public AttentionBlock(long embedDim, long heads, bool group) : base(nameof(AttentionBlock))
    {
        _group = group;
        norm = new RmsNorm(embedDim);
        attention = new SelfAttention(embedDim, heads);
        RegisterComponents();
    }

    public Tensor Forward(Tensor x, Tensor mask, (Tensor, Tensor, Tensor)? rope)
    {
        using var s = NewDisposeScope();
        var h = norm.forward(x);
        var o = _group
            ? attention.Forward(h.transpose(0, 1), mask, null).transpose(0, 1)
            : attention.Forward(h, mask, rope);
        return (x + o).MoveToOuterDisposeScope();
    }
}

/// <summary>SwiGLU with the gate in the FIRST half — the xFormers ordering, the reverse of
/// the usual.</summary>
public sealed class SwiGlu() : Module<Tensor, Tensor>(nameof(SwiGlu))
{
    public override Tensor forward(Tensor x)
    {
        using var s = NewDisposeScope();
        var halves = x.chunk(2, -1);
        return (functional.silu(halves[0]) * halves[1]).MoveToOuterDisposeScope();
    }
}

/// <summary>One attention block — time or group — then a pre-norm SwiGLU feed-forward.</summary>
public sealed class TransformerLayer : Module
{
    public bool IsGroup { get; }

    public AttentionBlock attention_block;
    public RmsNorm norm;
    public Sequential mlp;

    public TransformerLayer(long embedDim, long heads, long hidden, bool group) : base(nameof(TransformerLayer))
    {
        IsGroup = group;
        attention_block = new AttentionBlock(embedDim, heads, group);
        norm = new RmsNorm(embedDim);
        // Named as the checkpoint's nn.Sequential indexes them: mlp.0 and mlp.2.
        mlp = Sequential(
            ("0", Linear(embedDim, 2 * hidden)),
            ("1", new SwiGlu()),
            ("2", Linear(hidden, embedDim)));
        RegisterComponents();
    }

    public Tensor Forward(Tensor x, Tensor timeMask, Tensor groupMask, (Tensor, Tensor, Tensor) rope)
    {
        using var s = NewDisposeScope();
        var h = IsGroup
            ? attention_block.Forward(x, groupMask, null)
            : attention_block.Forward(x, timeMask, rope);
        return (h + mlp.forward(norm.forward(h))).MoveToOuterDisposeScope();
    }
}

/// <summary>
/// The layer stack: every <c>group_every_n</c>-th layer attends across rows, the rest along
/// time. For the published checkpoints that is two time layers then one group layer, eight
/// times over.
/// </summary>
public sealed class Transformer : Module
{
    private readonly XposRotary _rope;

    public ModuleList<TransformerLayer> layers;
    public RmsNorm out_norm;

    public Transformer(T0Config cfg) : base(nameof(Transformer))
    {
        _rope = new XposRotary(cfg.EmbedDim / cfg.NumHeads, cfg.XposHalvesLayout);
        layers = new ModuleList<TransformerLayer>(
            [.. Enumerable.Range(0, cfg.NumLayers).Select(i => new TransformerLayer(
                cfg.EmbedDim, cfg.NumHeads, cfg.MlpHiddenDim,
                group: cfg.GroupEveryN > 0 && (i + 1) % cfg.GroupEveryN == 0))]);
        out_norm = new RmsNorm(cfg.EmbedDim);
        RegisterComponents();
    }

    public (Tensor, Tensor, Tensor) Rope(long patches, Device device) => _rope.Tables(patches, device);

    /// <param name="x">Patch embeddings (rows, patches, embed).</param>
    /// <param name="timeMask">From <see cref="T0Preprocess.AttentionMasks"/>.</param>
    /// <param name="groupMask">From <see cref="T0Preprocess.AttentionMasks"/>.</param>
    public Tensor Forward(Tensor x, Tensor timeMask, Tensor groupMask)
    {
        using var s = NewDisposeScope();
        var rope = Rope(x.shape[1], x.device);
        var h = x;
        foreach (var layer in layers) h = layer.Forward(h, timeMask, groupMask, rope);
        return out_norm.forward(h).MoveToOuterDisposeScope();
    }
}

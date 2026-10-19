using System.Reflection;

using Tsfm.Forecasting;

namespace Tsfm.Forecasting.Timestral.Weights;

/// <summary>
/// Timestral-5M (5.2M), embedded. Context up to 32768 steps; 1024 per decoded block.
///
/// <para>Pinned revision <c>bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f</c>, embedded rather than fetched, so the weights a build
/// was compiled against cannot drift underneath it. The build refuses a payload whose
/// SHA-256 differs from the revision's.</para>
///
/// <para><b>Not open source.</b> Mistral AI's Modified MIT licence (<see cref="License"/>)
/// grants no rights to anyone whose company, or employer, had consolidated monthly revenue
/// above USD 20 million in the preceding month.</para>
/// </summary>
public sealed class Timestral5M : EmbeddedWeights
{
    public static readonly Timestral5M Instance = new();

    public override string Name => "mistralai/Timestral-5M@bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f";

    /// <summary>The weights' licence, verbatim as Mistral AI publishes it.</summary>
    public static string License
    {
        get
        {
            using var reader = new StreamReader(typeof(Timestral5M).Assembly.GetManifestResourceStream("timestral.license")!);
            return reader.ReadToEnd();
        }
    }

    protected override Assembly Host => typeof(Timestral5M).Assembly;
    protected override string ModelResource => "timestral.model";
    protected override string ModelConfigResource => "timestral.config";
}

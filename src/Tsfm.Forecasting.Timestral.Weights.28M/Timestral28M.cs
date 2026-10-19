using System.Reflection;

using Tsfm.Forecasting;

namespace Tsfm.Forecasting.Timestral.Weights;

/// <summary>
/// Timestral-28M (27.8M), embedded. Context up to 32768 steps; 1024 per decoded block.
///
/// <para>Pinned revision <c>180698bbf5a50dda3516aa6419798437fb03a430</c>, embedded rather than fetched, so the weights a build
/// was compiled against cannot drift underneath it. The build refuses a payload whose
/// SHA-256 differs from the revision's.</para>
///
/// <para><b>Not open source.</b> Mistral AI's Modified MIT licence (<see cref="License"/>)
/// grants no rights to anyone whose company, or employer, had consolidated monthly revenue
/// above USD 20 million in the preceding month.</para>
/// </summary>
public sealed class Timestral28M : EmbeddedWeights
{
    public static readonly Timestral28M Instance = new();

    public override string Name => "mistralai/Timestral-28M@180698bbf5a50dda3516aa6419798437fb03a430";

    /// <summary>The weights' licence, verbatim as Mistral AI publishes it.</summary>
    public static string License
    {
        get
        {
            using var reader = new StreamReader(typeof(Timestral28M).Assembly.GetManifestResourceStream("timestral.license")!);
            return reader.ReadToEnd();
        }
    }

    protected override Assembly Host => typeof(Timestral28M).Assembly;
    protected override string ModelResource => "timestral.model";
    protected override string ModelConfigResource => "timestral.config";
}

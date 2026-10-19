namespace Tsfm.Forecasting;

/// <summary>
/// A single-network checkpoint: <c>config.json</c> beside <c>model.safetensors</c>. The
/// tokenizer-free counterpart of <see cref="ICheckpoint"/>, for models that read values
/// directly. Streams, not paths: an embedded resource is located by the runtime rather
/// than by configuration.
/// </summary>
public interface IModelWeights
{
    /// <summary>Repository name and revision, for provenance.</summary>
    string Name { get; }

    /// <summary>The safetensors payload.</summary>
    Stream OpenModel();

    /// <summary>The published <c>config.json</c>, verbatim.</summary>
    string ModelConfigJson { get; }
}

/// <summary>Weights from a snapshot directory holding <c>config.json</c> and
/// <c>model.safetensors</c>, as published.</summary>
public sealed class DirectoryWeights : IModelWeights
{
    private readonly string _dir;

    /// <param name="dir">The snapshot directory.</param>
    /// <param name="name">Provenance label; defaults to the directory name.</param>
    /// <param name="howToFetch">Appended to the error when a file is missing, so the
    /// message says where the weights come from.</param>
    public DirectoryWeights(string dir, string? name = null, string? howToFetch = null)
    {
        _dir = dir;
        Name = name ?? Path.GetFileName(dir.TrimEnd(Path.DirectorySeparatorChar));
        foreach (var file in (string[])["config.json", "model.safetensors"])
            if (!File.Exists(Path.Combine(dir, file)))
                throw new FileNotFoundException(
                    $"{Path.Combine(dir, file)} not found. {howToFetch}".TrimEnd(), Path.Combine(dir, file));
    }

    public string Name { get; }

    public override string ToString() => Name;

    public Stream OpenModel() => File.OpenRead(Path.Combine(_dir, "model.safetensors"));

    public string ModelConfigJson => File.ReadAllText(Path.Combine(_dir, "config.json"));
}

/// <summary>Weights carried as embedded resources. Derived types supply the assembly and
/// resource names.</summary>
public abstract class EmbeddedWeights : IModelWeights
{
    protected abstract System.Reflection.Assembly Host { get; }
    protected abstract string ModelResource { get; }
    protected abstract string ModelConfigResource { get; }

    public abstract string Name { get; }

    public override string ToString() => Name;

    public Stream OpenModel() => Open(ModelResource);

    public string ModelConfigJson
    {
        get
        {
            using var reader = new StreamReader(Open(ModelConfigResource));
            return reader.ReadToEnd();
        }
    }

    private Stream Open(string resource)
        => Host.GetManifestResourceStream(resource)
           ?? throw new InvalidOperationException(
               $"resource '{resource}' missing from {Host.GetName().Name}; " +
               "the weights assembly was built without its payload");
}

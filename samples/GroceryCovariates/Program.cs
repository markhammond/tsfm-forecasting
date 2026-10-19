// The grocery-store toy from Google's TimesFM covariates notebook: forecast next week's
// daily sales for ice cream and sunscreen, given this week's sales plus temperature and
// promotion schedules that are known for both weeks.
//
// NOT a reproduction of that notebook's numbers. It uses forecast_with_covariates, the
// XReg path, which forecasts the base series and then regresses covariates on the
// residuals. TimesFM 3.0 instead attends over the covariates as variates, so the
// mechanism differs and the outputs should not be expected to match.
//
// Either model runs through the same IQuantileForecaster call; only the layout differs,
// because TimesFM reads the known future by rolling whole patches and t0 does not.
//
//   dotnet run --project samples/GroceryCovariates              # TimesFM 3.0
//   dotnet run --project samples/GroceryCovariates -- t0-beta   # or t0-alpha
//   dotnet run --project samples/GroceryCovariates -- Timestral-5M   # embedded; or -28M, -123M, -209M

using TorchSharp;

using Tsfm.Forecasting;
using Tsfm.Forecasting.T0;
using Tsfm.Forecasting.Timestral;
using Tsfm.Forecasting.Timestral.Weights;
using Tsfm.Forecasting.TimesFm;

using static TorchSharp.torch;

const int observed = 7, horizon = 7;                 // one week seen, one week ahead

// Known for both weeks — the point of the example.
float[] temperature = [31.0f, 24.3f, 19.4f, 26.2f, 24.6f, 30.0f, 31.1f,
                       32.4f, 30.9f, 26.0f, 25.0f, 27.8f, 29.5f, 31.2f];
float[] weekday     = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6];

var products = new (string Name, float[] Sales, float[] Promotion)[]
{
    ("ice cream", [30, 30, 4, 5, 7, 8, 10], [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
    ("sunscreen", [5, 7, 12, 13, 5, 6, 10],  [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]),
};

Device device;
try { device = new Device(DeviceType.MPS); using (ones(1).to(device)) { } }
catch { device = new Device(DeviceType.CPU); }

var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
var model = args.Length > 0 ? args[0] : "timesfm-3.0-pytorch";
var checkpoint = Path.Combine(root, "checkpoints", model);

IQuantileForecaster forecaster;
int context, total, pad;
if (model.StartsWith("Timestral"))
{
    // Like t0, Timestral pads a short context itself. 5M and 28M are embedded; the larger
    // two load from a fetched directory.
    forecaster = model switch
    {
        "Timestral-5M" => TimestralForecaster.Load(Timestral5M.Instance, device),
        "Timestral-28M" => TimestralForecaster.Load(Timestral28M.Instance, device),
        _ => TimestralForecaster.Load(checkpoint, device),
    };
    (context, total, pad) = (observed, observed + horizon, 0);
    Console.WriteLine($"{model} on {device.type}: {observed} days observed, {horizon} forecast\n");
}
else if (model.StartsWith("t0"))
{
    // t0 left-pads a short context onto its patch grid itself, and masks the padding.
    forecaster = T0Forecaster.Load(checkpoint, device);
    (context, total, pad) = (observed, observed + horizon, 0);
    Console.WriteLine($"{model} on {device.type}: {observed} days observed, {horizon} forecast\n");
}
else
{
    // TimesFM patches 32 steps at a time and reads the known future by rolling whole
    // patches forward, so the context must fill one patch and the horizon must sit in the
    // next. Seven days is far below that granularity: the observed week is left-padded to
    // fill a patch, and the forecast week occupies the one after it.
    var timesFm = TimesFmForecaster.Load(checkpoint, device);
    var patch = timesFm.Config.InputPatchLen;
    (forecaster, context, total, pad) = (timesFm, patch, 2 * patch, patch - observed);
    Console.WriteLine($"{model} on {device.type}: {observed} days observed, {horizon} forecast; "
                      + $"padded to {context} context + {patch} horizon patch (patch length {patch})\n");
}

var quantiles = forecaster.Quantiles;
var (lo, mid, hi) = (Level(0.1), Level(0.5), Level(0.9));

foreach (var (name, sales, promotion) in products)
{
    // Variate 0 is the target; the rest are known through the horizon. Order is
    // arbitrary — the model has no variate identity and infers relations from values.
    const int variates = 4;
    var series = new float[variates * total];
    for (var t = 0; t < observed + horizon; t++)
    {
        var i = pad + t;
        series[0 * total + i] = t < observed ? sales[t] : 0f;
        series[1 * total + i] = temperature[t];
        series[2 * total + i] = promotion[t];
        series[3 * total + i] = weekday[t];
    }

    var q = forecaster.ForecastJoint(series, variates, context, horizon,
        knownFuture: [false, true, true, true], targetVariate: 0);

    Console.WriteLine($"{name}  (observed: {string.Join(", ", sales)})");
    Console.WriteLine($"  day  promo  temp    p{quantiles[lo] * 100:F0}     p{quantiles[mid] * 100:F0}     p{quantiles[hi] * 100:F0}");
    for (var h = 0; h < horizon; h++)
        Console.WriteLine($"  {h + 1,3}  {(promotion[observed + h] > 0 ? "yes" : "  -"),5}"
                          + $"  {temperature[observed + h],4:F1}"
                          + $"  {q[h, lo],6:F1}  {q[h, mid],6:F1}  {q[h, hi],6:F1}");
    Console.WriteLine();
}

// The column nearest a level: the grids differ (TimesFM 9, t0-alpha 5, t0-beta and Timestral 21).
int Level(double level) => Enumerable.Range(0, quantiles.Count).MinBy(i => Math.Abs(quantiles[i] - level));

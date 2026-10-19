#!/usr/bin/env bash
# Build the full set of NuGet packages into one directory, verified and ready to upload.
#
# Fetches the checkpoints the weights packages embed (idempotent: files already present and
# matching their pinned revision are kept), packs every packable project at the version in
# Directory.Build.props, verifies the result, and prints the order to upload in.
#
#   ./scripts/pack.sh               # into artifacts/
#   ./scripts/pack.sh out/0.2.0     # or elsewhere
#
# Nothing is pushed. A version on nuget.org can be unlisted but never replaced, so upload
# the files this produced — not a later rebuild.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/artifacts}" && cd "${1:-$ROOT/artifacts}" && pwd)"
cd "$ROOT"

version=$(grep -o '<Version>[^<]*' Directory.Build.props | cut -d'>' -f2)
echo "Tsfm.Forecasting $version -> $OUT"

echo; echo "== checkpoints"
./scripts/fetch-checkpoints.sh
./scripts/fetch-timestral-checkpoint.sh

# A stale package of another version left in the directory would be uploaded by mistake.
echo; echo "== pack"
rm -f "$OUT"/*.nupkg "$OUT"/*.snupkg
dotnet pack -c Release --nologo -o "$OUT"

echo; echo "== verify"
./scripts/verify-packages.sh "$OUT"

# Each package depends on the exact version of its parent, and nuget.org takes minutes to
# index an upload, so parents must be uploaded (and indexed) before their dependants.
order=(
  Tsfm.Forecasting
  Tsfm.Forecasting.Kronos
  Tsfm.Forecasting.TimesFm
  Tsfm.Forecasting.T0
  Tsfm.Forecasting.Timestral
  Tsfm.Forecasting.Kronos.Weights.Mini
  Tsfm.Forecasting.Kronos.Weights.Small
  Tsfm.Forecasting.Timestral.Weights.5M
  Tsfm.Forecasting.Timestral.Weights.28M
)

echo; echo "== upload in this order (wait for each tier to index before the next)"
tier=1
for id in "${order[@]}"; do
  case "$id" in
    Tsfm.Forecasting.Kronos) tier=2 ;;
    Tsfm.Forecasting.Kronos.Weights.Mini) tier=3 ;;
  esac
  pkg="$OUT/$id.$version.nupkg"
  [ -f "$pkg" ] || { echo "missing $pkg"; exit 1; }
  sym="$OUT/$id.$version.snupkg"
  printf "  %d  %-58s %s\n" "$tier" "$(basename "$pkg")" "$([ -f "$sym" ] && basename "$sym")"
done

# Every package produced must be in the list, or it would never be uploaded.
produced=$(cd "$OUT" && ls *.nupkg | sed "s/\.$version\.nupkg$//" | sort)
listed=$(printf '%s\n' "${order[@]}" | sort)
if [ "$produced" != "$listed" ]; then
  echo; echo "the packages produced do not match the upload list:"
  diff <(echo "$listed") <(echo "$produced") || true
  exit 1
fi

echo
echo "Tsfm.Forecasting.Timestral.Weights.* are under Mistral AI's Modified MIT licence: no rights"
echo "for anyone whose company or employer exceeds USD 20M monthly revenue — publisher included."

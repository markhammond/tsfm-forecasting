#!/usr/bin/env bash
# Fetch the t0 checkpoints: t0-alpha (~407 MB) and t0-beta (~1.0 GB).
#
# The weights are Apache-2.0, published by The Forecasting Company. They are not bundled
# into a NuGet package only because both exceed nuget.org's 250 MB limit.
#
# Revisions are pinned rather than tracking main, and each payload is checked against its
# LFS SHA-256: a silently refreshed checkpoint would change every downstream forecast while
# leaving the consuming code identical, and a truncated one loads until a read runs off
# the end.
#
#   ./scripts/fetch-t0-checkpoint.sh              # both
#   ./scripts/fetch-t0-checkpoint.sh t0-alpha     # one
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
# macOS ships shasum, not sha256sum.
if command -v sha256sum > /dev/null; then SHA256="sha256sum"; else SHA256="shasum -a 256"; fi

fetch () {
  local name=$1 rev=$2 sha=$3
  local dir="$ROOT/$name"
  local base="https://huggingface.co/theforecastingcompany/$name/resolve/$rev"
  mkdir -p "$dir"
  for f in config.json LICENSE NOTICE README.md model.safetensors; do
    if [ -s "$dir/$f" ]; then echo "  have  $name/$f"; continue; fi
    echo "  fetch $name/$f"
    curl -fsSL --retry 3 -o "$dir/$f.part" "$base/$f"
    mv "$dir/$f.part" "$dir/$f"
  done
  echo "$sha  $dir/model.safetensors" | $SHA256 -c --quiet - \
    || { echo "  $name/model.safetensors does not match the pinned revision $rev"; exit 1; }
  echo "  ok    $name @ ${rev:0:12}"
}

want="${1:-all}"
[[ $want == all || $want == t0-alpha ]] && fetch t0-alpha \
  741927847fe02367624749bb9290b2f09f128e7d 16c030d3fd70f06dc4238e9a8356e9b5a631d07f80f1bc76ba539991aed5897f
[[ $want == all || $want == t0-beta ]] && fetch t0-beta \
  941e522547ec63f0f2aad57089fa0f5b8498bdf8 a0fd8abd51275dd30afd21f4892c763ebd45e2cb1811eff26eed43f89a543f7d
exit 0

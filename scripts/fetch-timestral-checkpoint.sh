#!/usr/bin/env bash
# Fetch the Timestral checkpoints: 5M (21 MB), 28M (111 MB), 123M (490 MB), 209M (837 MB).
#
# THE WEIGHTS ARE NOT OPEN SOURCE IN THE OSI SENSE. Mistral AI publishes them under a
# Modified MIT licence that withholds every right from anyone whose company (or employer)
# had consolidated monthly revenue above USD 20 million in the preceding month. The
# Apache-2.0 code in this repository grants nothing in the weights. Read the LICENSE file
# fetched beside each checkpoint before using it.
#
# 5M and 28M are also embedded in the Tsfm.Forecasting.Timestral.Weights.* packages; their
# builds read from here. Revisions are pinned and payloads checked against their SHA-256.
#
#   ./scripts/fetch-timestral-checkpoint.sh                  # 5M and 28M, the embedded pair
#   ./scripts/fetch-timestral-checkpoint.sh all              # all four
#   ./scripts/fetch-timestral-checkpoint.sh Timestral-209M   # one
set -euo pipefail

ROOT="${TIMESTRAL_CHECKPOINT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"
# macOS ships shasum, not sha256sum.
if command -v sha256sum > /dev/null; then SHA256="sha256sum"; else SHA256="shasum -a 256"; fi

fetch () {
  local name=$1 rev=$2 sha=$3
  local dir="$ROOT/$name"
  local base="https://huggingface.co/mistralai/$name/resolve/$rev"
  mkdir -p "$dir"
  for f in config.json LICENSE README.md model.safetensors; do
    if [ -s "$dir/$f" ]; then echo "  have  $name/$f"; continue; fi
    echo "  fetch $name/$f"
    curl -fsSL --retry 3 -o "$dir/$f.part" "$base/$f"
    mv "$dir/$f.part" "$dir/$f"
  done
  echo "$sha  $dir/model.safetensors" | $SHA256 -c --quiet - \
    || { echo "  $name/model.safetensors does not match the pinned revision $rev"; exit 1; }
  echo "  ok    $name @ ${rev:0:12}"
}

want="${1:-embedded}"
pick () { [[ $want == all || $want == "$1" || ($want == embedded && $2 == yes) ]]; }

echo "Timestral weights: Modified MIT, revenue-restricted. See LICENSE beside each checkpoint."
pick Timestral-5M yes && fetch Timestral-5M \
  bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f 7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738
pick Timestral-28M yes && fetch Timestral-28M \
  180698bbf5a50dda3516aa6419798437fb03a430 dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d
pick Timestral-123M no && fetch Timestral-123M \
  5a147287addda85f6acd8da3fb8858fe9af7e8f7 599fafa73cadbbc800b5a346ca900fa741ee3e5fd22e3b4a4430e151e74f4bc2
pick Timestral-209M no && fetch Timestral-209M \
  9874d252ab0aae16f7f7c999f98f98d6ad7d8365 1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa
exit 0

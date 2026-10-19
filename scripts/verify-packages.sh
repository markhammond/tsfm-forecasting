#!/usr/bin/env bash
# Check the packages carry what they must before anything is pushed. A push to nuget.org
# is irreversible — a version can be delisted but never replaced — so the licence and
# attribution are verified here rather than discovered missing afterwards.
set -euo pipefail

DIR="${1:-artifacts}"
fail=0

for pkg in "$DIR"/*.nupkg; do
  [ -e "$pkg" ] || { echo "no packages in $DIR"; exit 1; }
  name=$(basename "$pkg")
  size=$(( $(wc -c < "$pkg") / 1048576 ))
  echo "── $name (${size} MB)"

  contents=$(unzip -l "$pkg")
  required_files="LICENSE README.md"
  # Revenue-restricted weights must carry their own licence, and demand acceptance of it.
  case "$name" in
    Tsfm.Forecasting.Timestral.Weights.*)
      required_files="$required_files LICENSE-Timestral-Weights"
      if unzip -p "$pkg" '*.nuspec' | grep -q '<requireLicenseAcceptance>true'; then
        echo "   ok      requires licence acceptance"
      else
        echo "   MISSING requireLicenseAcceptance"; fail=1
      fi ;;
  esac
  for required in $required_files; do
    if grep -q "$required" <<< "$contents"; then
      echo "   ok      carries $required"
    else
      echo "   MISSING $required"; fail=1
    fi
  done

  # nuget.org rejects anything larger than 250 MB; failing here beats failing on push.
  if [ "$size" -ge 250 ]; then
    echo "   TOO BIG ${size} MB exceeds the 250 MB nuget.org limit"; fail=1
  fi

  # A weights package with no embedded payload builds and publishes cleanly, then fails
  # at load for every consumer — worth an explicit check.
  case "$name" in
    Tsfm.Forecasting.Kronos.Weights.*|Tsfm.Forecasting.Timestral.Weights.*)
      if [ "$size" -lt 10 ]; then
        echo "   EMPTY   weights package is only ${size} MB; the checkpoint did not embed"; fail=1
      else
        echo "   ok      payload present"
      fi ;;
  esac
done

exit "$fail"

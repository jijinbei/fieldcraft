#!/usr/bin/env bash
# Runs the exchange-snapshot experiment from the repository root:
#   SOLIDCRAFT=/path/to/solidcraft experiments/exchange-snapshot/run.sh
# SOLIDCRAFT is a checkout of the branch with `solidcraft-exchange`
# (https://github.com/tangency-labs/solidcraft/pull/8845).
set -euo pipefail
: "${SOLIDCRAFT:?set SOLIDCRAFT to a SolidCraft checkout with export-snapshot}"
here=experiments/exchange-snapshot
out=target/exchange-snapshot
design=tests/oracle/filleted-bracket-fillet-stress-coarse/design.json
mkdir -p "$out"

# The same design two ways: FieldCraft's own build of it, and SolidCraft's snapshot.
cargo +stable run --release -q -p fieldcraft-geometry --features scripts --example compile_recipe -- \
  "$design" "$out/bracket.solidcraft"
(cd "$SOLIDCRAFT" && cargo +stable run --release -q -p solidcraft-cli -- run "$OLDPWD/$design" \
  --save "$OLDPWD/$out/bracket.sc" --quiet)
(cd "$SOLIDCRAFT" && cargo +stable run --release -q -p solidcraft-cli -- export-snapshot \
  "$OLDPWD/$out/bracket.sc" --out "$OLDPWD/$out/bracket-snapshot.json" --tolerance 0.088)

cargo +stable build --release -q -p fieldcraft-engine --example replies
for level in coarse fine; do
  for path in native snapshot; do
    echo "== $level / $path"
    target/release/examples/replies "$here/$level-$path.json" | grep -E '^[0-9]+ (mesh.tet.generate|solve.probe) '
  done
done

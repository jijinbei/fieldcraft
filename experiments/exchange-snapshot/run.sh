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

# Every oracle case built from a SolidCraft design, opened directly and from its snapshot.
cases="cdt-cylinder:--tolerance=0.05 cdt-stepped-block:--tolerance=0.05
  filleted-bracket-fillet-stress-coarse:--tolerance=0.088 filleted-bracket-fillet-stress:--tolerance=0.088
  octree-filleted-bracket:--relative-tolerance=0.001 conjugate-cavity-centred-body:--relative-tolerance=0.001"
mkdir -p "$out/cases"
names=()
for spec in $cases; do
  c=${spec%%:*}; t=${spec#*:}
  (cd "$SOLIDCRAFT" && cargo +stable run --release -q -p solidcraft-cli -- run "$OLDPWD/tests/oracle/$c/design.json" \
    --save "$OLDPWD/$out/cases/$c.sc" --quiet)
  (cd "$SOLIDCRAFT" && cargo +stable run --release -q -p solidcraft-cli -- export-snapshot \
    "$OLDPWD/$out/cases/$c.sc" --out "$OLDPWD/$out/cases/$c.json" "${t%%=*}" "${t#*=}")
  names+=("$c")
done
cargo +stable run --release -q -p fieldcraft-testkit --example exchange_compare -- "$out/cases" "${names[@]}"

# The mesher's own spread: the bracket's snapshot with every coordinate moved by up to 1e-7 mm.
cargo +stable build --release -q -p fieldcraft-testkit --example perturb_snapshot
for seed in 1 2 3 4 5; do
  target/release/examples/perturb_snapshot "$out/bracket-snapshot.json" "$out/perturbed.json" "$seed" 1e-7
  for level in coarse fine; do
    sed "s#$out/bracket-snapshot.json#$out/perturbed.json#" "$here/$level-snapshot.json" > "$out/perturbed-$level.json"
    echo "seed $seed $level: $(target/release/examples/replies "$out/perturbed-$level.json" | grep -oE '"max_von_mises":\{"node":[0-9]+,"value":[0-9.]+' | grep -oE '[0-9.]+$' | head -1)"
  done
done

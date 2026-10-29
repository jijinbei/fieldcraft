# Experiment: FieldCraft reading SolidCraft exchange snapshots

**Not for merging.** This branch is the downstream check for the SolidCraft pull request
[tangency-labs/solidcraft#8845](https://github.com/tangency-labs/solidcraft/pull/8845)
(issue [#8844](https://github.com/tangency-labs/solidcraft/issues/8844)), which adds
`solidcraft-exchange`: a design evaluated at some parameter values, written as plain, versioned
JSON with named faces, edges and corners, their exact surfaces and curves, and the feature
timeline.

The question: can FieldCraft get the same results from that JSON alone as from opening the
`.solidcraft` design through four SolidCraft crates, as it does today?

## What changed in FieldCraft

- `crates/geometry/src/snapshot.rs`: a serde-only reader of the snapshot (no SolidCraft crate).
  Each body becomes a `TaggedSurface` with its face names and each face's exact surface.
- `geometry.import.mesh` reads a file starting with `{` as a snapshot.
- `mesh.tet.generate` and the other meshers use the snapshot's exact surfaces for curved faces
  (boundary and mid-edge nodes on the true cylinder), as they do for designs opened directly.
- `crates/engine/examples/replies.rs`: runs a script and prints each command's reply.
- `crates/testkit`: `oracle::run_recipe` (the tail of `run_case`, for a session that already holds
  its geometry), `examples/exchange_compare.rs` (runs oracle cases both ways) and
  `examples/perturb_snapshot.rs` (moves a snapshot's vertices by a tiny pseudo-random amount).

## Case

The oracle case `filleted-bracket-fillet-stress` (and `-coarse`): an 8 mm steel L-bracket with an
R6 inside fillet, clamped at the foot, 2 kN on the top face. Expected: peak von Mises on the
fillet 213 MPa ±5%. The recipes here are the oracle's recipe with the geometry step swapped, plus
a `solve.probe` on the fillet:

| file | geometry |
|---|---|
| `*-native.json` | `geometry.open` of the design built by FieldCraft (today's path) |
| `*-snapshot.json` | `geometry.import.mesh` of the snapshot exported by SolidCraft |

Faces are selected by the same persistent names in both (`F2:side:l1`, `F2:side:l5`,
`F3:blend:F2:side:l3|F2:side:l4`).

## Results

### Every oracle case built from a SolidCraft design

The six oracle cases whose geometry is a SolidCraft design (`design.json`), each run as the oracle
runs it and again with the design replaced by its snapshot (`crates/testkit/examples/exchange_compare.rs`;
`mesh.*` steps that name a body name the imported surface instead). Every quantity passes its
reference both ways.

| case | what it checks | snapshot vs direct |
|---|---|---|
| `cdt-cylinder` | tet mesh volume of a cylinder | 1.3e-12 |
| `cdt-stepped-block` | tet mesh volume, validity | 3e-16 |
| `octree-filleted-bracket` | hex octree mesh, stresses at three sizes | identical |
| `conjugate-cavity-centred-body` | conjugate heat transfer (fluid + solid), Nusselt numbers | identical |
| `filleted-bracket-fillet-stress-coarse` | tet10 peak stress on the R6 fillet | −1.3% |
| `filleted-bracket-fillet-stress` | the same, fine | −0.36% |

### The filleted bracket in detail

| path | coarse: tets | coarse: peak | fine: tets | fine: peak |
|---|---|---|---|---|
| opening `.solidcraft` directly | 3107 | 211.05 MPa | 11497 | 213.61 MPa |
| exchange snapshot | 3132 | 208.30 MPa | 11507 | 212.84 MPa |
| face-tagged 3MF (no exact surfaces) | 9179 | 211.8 MPa | | 213.0 MPa |

Reference 213 MPa ±5% (202.35–223.65 MPa).

- Both the direct path and the snapshot give the same 544 surface triangles, face by face.
- The difference is within the tet mesher's own spread. Moving every snapshot coordinate by a
  pseudo-random amount up to 1e-7 mm (`perturb_snapshot`, seeds 1–5) moves the peak over:
  - coarse: 202.3–211.1 MPa (seed 1 is 0.1 MPa above the oracle's lower bound);
  - fine: 212.3–214.9 MPa.
  The snapshot's 208.30 / 212.84 MPa and the direct path's 211.05 / 213.61 MPa all lie inside.
  The coarse level's sensitivity to 1e-7 mm is FieldCraft's own, not the snapshot's.
- The 3MF path has no exact surfaces, so the mesher refines the fillet itself: 3× the elements
  and about 4× the time for the same answer.
- Defeaturing (suppressing a feature) and partitioning by a plane through the snapshot's edits give
  the same cells and face sets as FieldCraft's own operations.

## Running it

```sh
SOLIDCRAFT=/path/to/solidcraft experiments/exchange-snapshot/run.sh
```

`SOLIDCRAFT` is a checkout of the pull request's branch. The script builds the design both ways
into `target/exchange-snapshot/` and runs the four bracket recipes, then the six oracle cases both
ways and the five perturbed runs. Build with `cargo +stable` (the
`branches` crate SolidCraft depends on does not build on current nightly).

## The other downstream tool: EMCraft

EMCraft is an open-source electromagnetic field simulator written in Rust, being developed by
the same author and not yet published. It models microwave components (waveguides, filters,
antennas), solves Maxwell's equations with its own finite element solvers, and reports
S-parameters and fields. It sits beside FieldCraft and SolidCraft, with the same conventions: every
action is a command, nothing panics, and every solver result is checked against closed-form
solutions or published benchmarks.

EMCraft imports SolidCraft geometry and, without the snapshot, has to fit planes and cylinders
back onto 3MF triangles. Through the snapshot (a metal post in a WR-90 waveguide at 10 GHz,
against the closed-form reference):

| path | \|S11\| error | phase error |
|---|---|---|
| 3MF + surface fitting | 5.6e-5 | 0.020° |
| exchange snapshot | 5.0e-5 | 0.018° |

Same accuracy, because fitting already recovers the cylinder. What is new is a sweep of a design
parameter from one file, with no re-export: post diameter 2.5 / 3 / 4 mm gives |S11| errors
6.6e-5 / 5.0e-5 / 2.9e-5, phase within 0.019°.

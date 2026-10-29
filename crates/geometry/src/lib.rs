//! The geometry boundary.
//!
//! Meshing in FieldCraft starts from a triangulated boundary surface. This crate produces it:
//! designs are opened and evaluated by SolidCraft through its exchange crate
//! (`solidcraft-exchange`: plain, typed snapshots of a design's bodies; plan/adr/0001-libraries.md)
//! and leave it as a [`SurfaceMesh`] from SolidCraft's tessellator, with each triangle tagged by
//! the B-rep face it lies on (for feature edges and for loads and boundary conditions on faces).
//! No SolidCraft type appears in the public API. Only this crate may depend on `solidcraft-*`
//! (`cargo xtask layers` enforces it); without the `scripts` feature it depends on the exchange
//! crate alone.
//!
//! Geometry enters two ways: shapes built in code (such as [`block`]) and SolidCraft designs
//! opened with [`open_design`], which rebuilds the design's parametric timeline and gives its
//! bodies with persistent face names (a boundary condition on a named face follows that face
//! through an edit of the design).
//!
//! Lengths are millimetres.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

pub mod check;
pub mod defeature;
mod design;
mod edges;
pub mod face_split;
pub mod features;
pub mod partition;
#[cfg(any(test, feature = "scripts"))]
pub mod sample;
#[cfg(any(test, feature = "scripts"))]
mod script;
mod source;
pub mod stitch;
pub mod virtual_faces;

pub use design::{DESIGN_FORMAT, Design, FeatureError, MAX_DESIGN_BYTES, open_design};
pub use edges::{EDGE_TOLERANCE, Edge, Vertex};
pub use features::{BodyFeatures, Curve, FeatureNode, ProfileGeometry, SketchGeometry};
pub use partition::{Cut, CutReport};
pub use virtual_faces::Merge;
pub mod snapshot;
pub mod tagged_mesh;
#[cfg(any(test, feature = "scripts"))]
pub use script::{MAX_SCRIPT_BYTES, script_design};
use source::{Fixed, Source};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("the geometry operation failed: {0}")]
    Failed(String),
    /// A design saved by a SolidCraft newer than the one FieldCraft is built with
    /// ([`open_design`]): FieldCraft needs updating, not SolidCraft.
    #[error("this design was saved by a newer SolidCraft (format {format}); this FieldCraft reads up to {reads}. Update FieldCraft to open it")]
    NewerDesign { format: u32, reads: u32 },
    /// From the model layer: a cancelled job ([`fieldcraft_model::progress::check`]).
    #[error(transparent)]
    Model(#[from] fieldcraft_model::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<solidcraft_exchange::Error> for Error {
    fn from(e: solidcraft_exchange::Error) -> Self {
        source::error(e)
    }
}

/// A solid body.
#[derive(Debug, Clone)]
pub struct Shape {
    pub name: String,
    source: Source,
    /// By face index (the [`SurfaceMesh::tri_face`] tags); empty for shapes built in code. With
    /// merged faces ([`virtual_faces`]) these are the virtual faces, not the B-rep's.
    faces: Vec<Face>,
    /// The virtual face (index into `faces`) of each B-rep face; empty: they are the same.
    face_map: Vec<u32>,
    /// Faces split along a plane ([`face_split`]), applied after the merges, in order.
    cuts: Vec<face_split::Cut>,
}

/// A B-rep face's persistent identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Face {
    /// Stable across rebuilds of the design: named after the feature and sketch curve that made
    /// the face (`F2:side:l4`, `F2:top`, …), not after its position in the body.
    pub name: String,
    /// The name follows the history SolidCraft recorded; `false` when it was worked out from the
    /// geometry, which an upstream edit can fool.
    pub exact: bool,
    /// A merged face's B-rep faces, by persistent name ([`virtual_faces`]); empty for a B-rep face.
    pub members: Vec<String>,
}

/// The analytic surface a B-rep face lies on, when it is one of the elementary kinds. Directions
/// are unit vectors; lengths are millimetres and angles radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaceSurface {
    /// The outward unit normal and a point on the plane.
    Plane {
        normal: [f64; 3],
        point: [f64; 3],
    },
    Cylinder {
        origin: [f64; 3],
        axis: [f64; 3],
        radius: f64,
    },
    /// The apex, the axis pointing into the cone and the half-angle.
    Cone {
        apex: [f64; 3],
        axis: [f64; 3],
        half_angle: f64,
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    /// The centre of the ring, its axis, and the ring (major) and tube (minor) radii.
    Torus {
        center: [f64; 3],
        axis: [f64; 3],
        major: f64,
        minor: f64,
    },
}

/// Mass properties and extent of a [`Shape`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measure {
    pub volume: f64,
    pub area: f64,
    pub centroid: [f64; 3],
    pub bbox_min: [f64; 3],
    pub bbox_max: [f64; 3],
    pub faces: usize,
}

/// A triangulated boundary surface: the input to volume meshing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceMesh {
    pub positions: Vec<[f64; 3]>,
    /// Counter-clockwise seen from outside.
    pub triangles: Vec<[u32; 3]>,
    /// The B-rep face each triangle lies on.
    pub tri_face: Vec<u32>,
    /// Copies of one point closer than this are the same point (0: only identical coordinates).
    /// A [`Shape::tessellate`] result is a triangle soup whose faces each carry their own copy of
    /// the boundary points they share, accurate to the kernel's linear tolerance, so two copies
    /// can differ by up to twice it; boundary surface preparation welds at this distance.
    pub weld_distance: f64,
    /// The B-rep edges as polylines, by edge index (empty for a surface without B-rep edges).
    pub edges: Vec<Vec<[f64; 3]>>,
    /// Per edge: the B-rep faces it bounds (one or two), by face index.
    pub edge_faces: Vec<Vec<u32>>,
}

/// The largest coordinate a box built in code may have.
const MAX_BLOCK: f64 = 1e9; // unit-factors: a cap on hostile input

/// A box between two corners.
pub fn block(a: [f64; 3], b: [f64; 3]) -> Result<Shape> {
    let lo: [f64; 3] = std::array::from_fn(|k| a[k].min(b[k]));
    let hi: [f64; 3] = std::array::from_fn(|k| a[k].max(b[k]));
    let finite = a.iter().chain(&b).all(|v| v.is_finite() && v.abs() <= MAX_BLOCK);
    if !finite || (0..3).any(|k| !(hi[k] - lo[k] > source::LINEAR_TOLERANCE)) {
        return Err(Error::Invalid("a box needs finite corners apart along every axis".into()));
    }
    // Corner k has x from bit 0, y from bit 1, z from bit 2.
    let positions: Vec<[f64; 3]> = (0..8).map(|k: usize| std::array::from_fn(|i| if k >> i & 1 == 1 { hi[i] } else { lo[i] })).collect();
    // Faces -x, +x, -y, +y, -z, +z: corners counter-clockwise seen from outside.
    let quads: [([u32; 4], [f64; 3]); 6] = [
        ([0, 4, 6, 2], [-1.0, 0.0, 0.0]),
        ([1, 3, 7, 5], [1.0, 0.0, 0.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([2, 6, 7, 3], [0.0, 1.0, 0.0]),
        ([0, 2, 3, 1], [0.0, 0.0, -1.0]),
        ([4, 5, 7, 6], [0.0, 0.0, 1.0]),
    ];
    let mut triangles = Vec::with_capacity(12);
    let mut tri_face = Vec::with_capacity(12);
    for (f, (q, _)) in (0u32..).zip(&quads) {
        triangles.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
        tri_face.extend([f, f]);
    }
    let mut edges = Vec::with_capacity(12);
    let mut edge_faces = Vec::with_capacity(12);
    for (i, (p, _)) in quads.iter().enumerate() {
        for (j, (q, _)) in quads.iter().enumerate().skip(i + 1) {
            let shared: Vec<u32> = p.iter().copied().filter(|c| q.contains(c)).collect();
            if let [c, d] = shared[..] {
                let at = |c: u32| positions.get(c as usize).copied().unwrap_or_default();
                edges.push(vec![at(c), at(d)]);
                edge_faces.push(vec![i as u32, j as u32]);
            }
        }
    }
    let surfaces = quads
        .iter()
        .map(|(q, n)| Some(FaceSurface::Plane { normal: *n, point: positions.get(q[0] as usize).copied().unwrap_or_default() }))
        .collect();
    let e: [f64; 3] = std::array::from_fn(|k| hi[k] - lo[k]);
    let measure = Measure {
        volume: e[0] * e[1] * e[2],
        area: 2.0 * (e[0] * e[1] + e[1] * e[2] + e[0] * e[2]),
        centroid: std::array::from_fn(|k| 0.5 * (lo[k] + hi[k])),
        bbox_min: lo,
        bbox_max: hi,
        faces: 6,
    };
    let size = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
    let mesh = SurfaceMesh { positions, triangles, tri_face, weld_distance: 0.0, edges, edge_faces };
    let fixed = Fixed { mesh, surfaces, measure, size };
    Ok(Shape { name: "Block".into(), source: Source::Fixed(std::sync::Arc::new(fixed)), faces: Vec::new(), face_map: Vec::new(), cuts: Vec::new() })
}

impl Shape {
    /// The persistent identities of the faces, by B-rep face index (empty for shapes built in code).
    pub fn faces(&self) -> &[Face] {
        &self.faces
    }

    /// The index of the face with this persistent name: the [`SurfaceMesh::tri_face`] tag its
    /// triangles carry.
    pub fn face_index(&self, name: &str) -> Option<u32> {
        self.faces.iter().position(|f| f.name == name).and_then(|i| u32::try_from(i).ok())
    }

    /// The area centroid of the face with this persistent name, from a tessellation fine enough
    /// to place it (on a curved face the centroid lies off the surface, inside its curvature).
    pub fn face_centre(&self, name: &str) -> Result<[f64; 3]> {
        let face = self.face_index(name).ok_or_else(|| Error::Invalid(format!("body `{}` has no face `{name}`", self.name)))?;
        let mesh = self.tessellate((self.size()? * edges::EDGE_TOLERANCE).max(1e-4))?;
        let (mut sum, mut area) = ([0.0; 3], 0.0);
        for (t, _) in mesh.triangles.iter().zip(&mesh.tri_face).filter(|(_, f)| **f == face) {
            let corner = |i: u32| mesh.positions.get(i as usize).copied();
            let (Some(a), Some(b), Some(c)) = (corner(t[0]), corner(t[1]), corner(t[2])) else { continue };
            let (u, v) = ([0, 1, 2].map(|k| b[k] - a[k]), [0, 1, 2].map(|k| c[k] - a[k]));
            let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let w = 0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            for (k, s) in sum.iter_mut().enumerate() {
                *s += w * (a[k] + b[k] + c[k]) / 3.0;
            }
            area += w;
        }
        if area > 0.0 && area.is_finite() {
            Ok(sum.map(|s| s / area))
        } else {
            Err(Error::Failed(format!("face `{name}` of body `{}` has no area", self.name)))
        }
    }

    /// The analytic surface of each B-rep face, by face index (the [`SurfaceMesh::tri_face`]
    /// tags): `None` for a free-form face. Meshers project nodes onto these surfaces. A merged
    /// face has the surface its B-rep faces share, `None` when they lie on different ones.
    pub fn face_surfaces(&self) -> Result<Vec<Option<FaceSurface>>> {
        let brep = self.brep_face_surfaces()?;
        self.virtual_surfaces(brep)
    }

    /// [`Shape::face_surfaces`] by B-rep face index (the pieces of [`Shape::tessellate_pieces`]).
    pub fn brep_face_surfaces(&self) -> Result<Vec<Option<FaceSurface>>> {
        self.source.surfaces()
    }

    /// A characteristic size: the diagonal of the bounding box of the body's B-rep vertices
    /// (cached by the kernel; curved faces may bulge past it). For default tolerances and sizes.
    pub fn size(&self) -> Result<f64> {
        self.source.size()
    }

    /// The boundary shells of the body: one for a solid in one piece without voids.
    pub fn shells(&self) -> Result<usize> {
        self.source.shells()
    }

    pub fn measure(&self) -> Result<Measure> {
        let m = self.source.measure()?;
        Ok(Measure { faces: if self.face_map.is_empty() && self.cuts.is_empty() { m.faces } else { self.faces.len() }, ..m })
    }

    /// Triangulate the boundary within `tolerance` (chordal deviation, mm). Triangles are tagged
    /// by face index ([`Shape::faces`]): merged faces are one face, without the edges between
    /// their B-rep faces.
    pub fn tessellate(&self, tolerance: f64) -> Result<SurfaceMesh> {
        Ok(self.virtual_mesh(self.tessellate_brep(tolerance)?))
    }

    /// [`Shape::tessellate`], and per triangle the B-rep face it lies on: its piece of a merged
    /// face, whose surface is in [`Shape::brep_face_surfaces`] (the same as its tag when no face
    /// is merged).
    pub fn tessellate_pieces(&self, tolerance: f64) -> Result<(SurfaceMesh, Vec<u32>)> {
        let brep = self.tessellate_brep(tolerance)?;
        let pieces = brep.tri_face.clone();
        Ok((self.virtual_mesh(brep), pieces))
    }

    /// The B-rep's own tessellation, tagged by B-rep face.
    fn tessellate_brep(&self, tolerance: f64) -> Result<SurfaceMesh> {
        if !(tolerance > 0.0 && tolerance.is_finite()) {
            // Its value is in the geometry's millimetres, which the caller may not show: not named.
            return Err(Error::Invalid("the tessellation tolerance must be a finite, positive length".into()));
        }
        self.source.tessellate(tolerance)
    }

    /// Whether some face is made of several B-rep faces ([`virtual_faces`]).
    pub fn has_merged_faces(&self) -> bool {
        !self.face_map.is_empty()
    }
}

impl SurfaceMesh {
    /// The volume the triangles enclose (divergence theorem: the sum of the signed tetrahedra each
    /// triangle spans with the origin). Equals the solid's volume for a closed, outward-oriented
    /// surface; a triangle with a vertex index out of range counts as nothing.
    pub fn enclosed_volume(&self) -> f64 {
        let six_v: f64 = self
            .triangles
            .iter()
            .filter_map(|t| {
                let [a, b, c] = [self.positions.get(t[0] as usize)?, self.positions.get(t[1] as usize)?, self.positions.get(t[2] as usize)?];
                Some(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]))
            })
            .sum();
        six_v / 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plate's side face at x = 0 (40 × 30 × 10, the hole well inside) has its centre in its
    /// middle; an unknown face is an error.
    #[test]
    fn a_face_centre_is_its_area_centroid() {
        let d = open_design(&sample::plate_design(&sample::Plate::default()).unwrap()).unwrap();
        let body = &d.bodies[0];
        let surfaces = body.face_surfaces().unwrap();
        let side = surfaces
            .iter()
            .position(|s| matches!(s, Some(FaceSurface::Plane { normal, point }) if normal[0] < -0.5 && point[0].abs() < 1e-9))
            .unwrap();
        let c = body.face_centre(&body.faces()[side].name).unwrap();
        assert!((c[0]).abs() < 1e-9 && (c[1] - 15.0).abs() < 1e-6 && (c[2] - 5.0).abs() < 1e-6, "{c:?}");
        assert!(body.face_centre("nope").is_err());
    }

    #[test]
    fn block_measures() {
        let m = block([0., 0., 0.], [10., 20., 30.]).unwrap().measure().unwrap();
        assert!((m.volume - 6000.0).abs() < 1e-6, "{}", m.volume);
        assert!((m.area - 2200.0).abs() < 1e-6, "{}", m.area);
        assert_eq!(m.faces, 6);
        assert_eq!(m.bbox_max, [10., 20., 30.]);
    }

    #[test]
    fn block_faces_are_planes_through_the_box() {
        let surfaces = block([0., 0., 0.], [10., 20., 30.]).unwrap().face_surfaces().unwrap();
        assert_eq!(surfaces.len(), 6);
        for s in surfaces {
            let Some(FaceSurface::Plane { normal, point }) = s else { panic!("{s:?}") };
            // Each plane holds one of the box's faces: the point's coordinate along the normal is
            // 0 or the box's extent there.
            let along: f64 = normal.iter().zip(point).map(|(n, p)| n * p).sum();
            assert!([0.0, 10.0, 20.0, 30.0].iter().any(|v| (along.abs() - v).abs() < 1e-9), "{normal:?} {point:?}");
        }
    }

    #[test]
    fn tessellation_covers_every_face_and_encloses_the_volume() {
        let mesh = block([0., 0., 0.], [2., 3., 4.]).unwrap().tessellate(0.01).unwrap();
        assert_eq!(mesh.triangles.len(), mesh.tri_face.len());
        let mut faces: Vec<u32> = mesh.tri_face.clone();
        faces.sort_unstable();
        faces.dedup();
        assert_eq!(faces.len(), 6);
        // The outward-oriented triangles enclose the block's volume.
        assert!((mesh.enclosed_volume() - 24.0).abs() < 1e-9, "{}", mesh.enclosed_volume());
    }

    #[test]
    fn hostile_input_is_an_error() {
        assert!(block([0.; 3], [0.; 3]).is_err());
        assert!(block([0.; 3], [f64::NAN, 1., 1.]).is_err());
        let b = block([0.; 3], [1.; 3]).unwrap();
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(b.tessellate(t).is_err());
        }
    }
}

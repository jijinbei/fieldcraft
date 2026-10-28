//! SolidCraft exchange snapshots (SolidCraft's `docs/formats/exchange-snapshot.md`, schema 1).
//!
//! EXPERIMENT: a design evaluated by SolidCraft and written as plain data — per body a welded
//! mesh whose faces carry their persistent names and exact surfaces. Read with serde alone, no
//! kernel call: each body becomes a [`TaggedSurface`] with its face names and surfaces.

use serde::Deserialize;

use crate::tagged_mesh::{MAX_ELEMENTS, TaggedSurface};
use crate::{Error, FaceSurface, Result, SurfaceMesh};

/// The newest schema read.
pub const SCHEMA: u32 = 1;

#[derive(Deserialize)]
struct Snapshot {
    schema: u32,
    unit: String,
    bodies: Vec<Body>,
}

#[derive(Deserialize)]
struct Body {
    name: String,
    closed: bool,
    positions: Vec<[f64; 3]>,
    faces: Vec<Face>,
    #[serde(default)]
    edges: Vec<SnapEdge>,
}

#[derive(Deserialize)]
struct SnapEdge {
    bounds: Bounds,
    #[serde(default)]
    path: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Bounds {
    Between(u32, u32),
    Seam(u32),
}

#[derive(Deserialize)]
struct Face {
    name: String,
    surface: Surface,
    triangles: Vec<[u32; 3]>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Surface {
    Plane {
        point: [f64; 3],
        normal: [f64; 3],
    },
    Cylinder {
        origin: [f64; 3],
        axis: [f64; 3],
        radius: f64,
    },
    Cone {
        apex: [f64; 3],
        axis: [f64; 3],
        half_angle: f64,
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    Torus {
        center: [f64; 3],
        axis: [f64; 3],
        major: f64,
        minor: f64,
    },
    #[serde(other)]
    Other,
}

fn invalid(msg: impl std::fmt::Display) -> Error {
    Error::Invalid(format!("snapshot: {msg}"))
}

/// The bodies of a snapshot, one surface each, in millimetres.
pub fn read_snapshot(bytes: &[u8]) -> Result<Vec<TaggedSurface>> {
    let s: Snapshot = serde_json::from_slice(bytes).map_err(invalid)?;
    if s.schema == 0 || s.schema > SCHEMA {
        return Err(invalid(format!("schema {} (this FieldCraft reads up to {SCHEMA})", s.schema)));
    }
    if s.unit != "mm" {
        return Err(invalid(format!("unit `{}` (mm expected)", s.unit)));
    }
    s.bodies.into_iter().map(body).collect()
}

fn body(b: Body) -> Result<TaggedSurface> {
    if !b.closed {
        return Err(invalid(format!("body `{}` is not closed", b.name)));
    }
    let n = b.positions.len();
    let total: usize = b.faces.iter().map(|f| f.triangles.len()).sum();
    if n > MAX_ELEMENTS || total > MAX_ELEMENTS {
        return Err(invalid(format!("body `{}` is too large", b.name)));
    }
    if b.positions.iter().flatten().any(|x| !x.is_finite() || x.abs() > fieldcraft_model::MAX_COORD) {
        return Err(invalid(format!("body `{}` has a coordinate out of range", b.name)));
    }
    let faces = u32::try_from(b.faces.len()).map_err(|_| invalid("too many faces"))?;
    let mut mesh = SurfaceMesh { positions: b.positions, ..SurfaceMesh::default() };
    let (mut face_names, mut surfaces) = (Vec::with_capacity(b.faces.len()), Vec::with_capacity(b.faces.len()));
    for (k, f) in b.faces.into_iter().enumerate() {
        if f.triangles.iter().flatten().any(|&v| v as usize >= n) {
            return Err(invalid(format!("body `{}` face `{}`: a vertex index is out of range", b.name, f.name)));
        }
        mesh.tri_face.extend(std::iter::repeat_n(k as u32, f.triangles.len()));
        mesh.triangles.extend(f.triangles);
        face_names.push(f.name);
        surfaces.push(match f.surface {
            Surface::Plane { point, normal } => Some(FaceSurface::Plane { normal, point }),
            Surface::Cylinder { origin, axis, radius } => Some(FaceSurface::Cylinder { origin, axis, radius }),
            Surface::Cone { apex, axis, half_angle } => Some(FaceSurface::Cone { apex, axis, half_angle }),
            Surface::Sphere { center, radius } => Some(FaceSurface::Sphere { center, radius }),
            Surface::Torus { center, axis, major, minor } => Some(FaceSurface::Torus { center, axis, major, minor }),
            Surface::Other => None,
        });
    }
    // The B-rep edges: along the mesh vertices that trace them when given, else their polylines.
    for e in &b.edges {
        let bounded = match e.bounds {
            Bounds::Between(a, c) => vec![a, c],
            Bounds::Seam(a) => vec![a],
        };
        if bounded.iter().any(|&f| f >= faces) {
            return Err(invalid(format!("body `{}`: an edge names a face out of range", b.name)));
        }
        let line: Option<Vec<[f64; 3]>> = e.path.iter().map(|&v| mesh.positions.get(v as usize).copied()).collect();
        let line = line.ok_or_else(|| invalid(format!("body `{}`: an edge's vertex is out of range", b.name)))?;
        mesh.edges.push(line);
        mesh.edge_faces.push(bounded);
    }
    Ok(TaggedSurface { name: b.name, mesh, faces, original: None, face_names, surfaces })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit tetrahedron: four planar faces, one triangle each, and one edge.
    const TETRA: &str = r#"{"schema": 1, "unit": "mm", "bodies": [{"name": "Tet", "closed": true,
        "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "faces": [
          {"name": "F1:-z", "surface": {"type": "plane", "point": [0, 0, 0], "normal": [0, 0, -1]}, "triangles": [[0, 2, 1]]},
          {"name": "F1:-y", "surface": {"type": "plane", "point": [0, 0, 0], "normal": [0, -1, 0]}, "triangles": [[0, 1, 3]]},
          {"name": "F1:-x", "surface": {"type": "plane", "point": [0, 0, 0], "normal": [-1, 0, 0]}, "triangles": [[0, 3, 2]]},
          {"name": "F1:xyz", "surface": {"type": "nurbs", "u": {"degree": 1, "knots": [0, 0, 1, 1]}}, "triangles": [[1, 2, 3]]}],
        "edges": [{"index": 0, "bounds": {"between": [0, 1]}, "path": [0, 1]}]}]}"#;

    #[test]
    fn a_snapshot_reads_as_named_faces_with_surfaces_and_edges() {
        let [t] = read_snapshot(TETRA.as_bytes()).unwrap().try_into().unwrap();
        assert_eq!((t.name.as_str(), t.faces, t.mesh.triangles.len()), ("Tet", 4, 4));
        assert_eq!(t.face_names, ["F1:-z", "F1:-y", "F1:-x", "F1:xyz"]);
        assert_eq!(t.mesh.tri_face, [0, 1, 2, 3]);
        assert!(matches!(t.surfaces[0], Some(FaceSurface::Plane { normal: [0.0, 0.0, -1.0], .. })));
        // A kind FieldCraft has no exact form for is meshed from its triangles.
        assert!(t.surfaces[3].is_none());
        assert_eq!((t.mesh.edges[0].clone(), t.mesh.edge_faces[0].clone()), (vec![[0.0; 3], [1.0, 0.0, 0.0]], vec![0, 1]));
    }

    #[test]
    fn hostile_snapshots_are_refused() {
        let bad = |from: &str, to: &str| read_snapshot(TETRA.replace(from, to).as_bytes());
        assert!(bad(r#""schema": 1"#, r#""schema": 9"#).is_err());
        assert!(bad(r#""unit": "mm""#, r#""unit": "in""#).is_err());
        assert!(bad(r#""closed": true"#, r#""closed": false"#).is_err());
        assert!(bad("[[1, 2, 3]]", "[[1, 2, 99]]").is_err());
        assert!(bad(r#"{"between": [0, 1]}"#, r#"{"between": [0, 9]}"#).is_err());
        assert!(bad(r#""path": [0, 1]"#, r#""path": [0, 99]"#).is_err());
        assert!(bad("[0, 0, 1]]", "[0, 0, 1e400]]").is_err());
        assert!(read_snapshot(b"not json").is_err());
    }
}

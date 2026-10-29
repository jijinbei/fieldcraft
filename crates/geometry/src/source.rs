//! Where a [`crate::Shape`]'s geometry comes from: a body of a design opened through
//! SolidCraft's exchange crate (re-evaluated at whatever tolerance is asked, with the design's
//! edits), or a fixed mesh for a shape built in code.
//!
//! Every query of a design body is a snapshot of that one body ([`solidcraft_exchange`]), kept
//! by tolerance so a second query at the same tolerance is free. No SolidCraft type leaves this
//! module except through the plain types of the crate's API.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use solidcraft_exchange::{Body, Edit, Options, Snapshot, Tolerance};

use crate::{Error, FaceSurface, Measure, Result, SurfaceMesh};

/// SolidCraft's linear modelling tolerance (mm): points closer than this are the same point.
pub(crate) const LINEAR_TOLERANCE: f64 = 1e-6;

/// The tolerance (a share of the body's size) the snapshot that answers questions about faces,
/// sizes and names is taken at.
pub(crate) const INFO_TOLERANCE: f64 = 1e-3;

/// Most snapshots kept per design.
const MAX_CACHED: usize = 256;

/// An opened design and the snapshots of its bodies taken so far.
pub(crate) struct DesignSource {
    design: solidcraft_exchange::Design,
    cache: Mutex<BTreeMap<String, Arc<Snapshot>>>,
}

impl std::fmt::Debug for DesignSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DesignSource").field("design", &self.design.name()).finish()
    }
}

/// Exchange errors as the crate's: a feature that fails or an edit the design refuses is the
/// geometry operation failing; a bad input is invalid.
pub(crate) fn error(e: solidcraft_exchange::Error) -> Error {
    match e {
        solidcraft_exchange::Error::Invalid(m) | solidcraft_exchange::Error::Read(m) => Error::Invalid(m),
        other => Error::Failed(other.to_string()),
    }
}

impl DesignSource {
    pub(crate) fn new(design: solidcraft_exchange::Design) -> Arc<DesignSource> {
        Arc::new(DesignSource { design, cache: Mutex::new(BTreeMap::new()) })
    }

    /// The snapshot with these edits and options, of `body` alone (all bodies: `None`).
    pub(crate) fn snapshot(&self, edits: &[Edit], body: Option<&str>, tolerance: Tolerance, mass: bool) -> Result<Arc<Snapshot>> {
        let key = format!("{body:?}\u{0}{tolerance:?}\u{0}{mass}\u{0}{}", serde_json::to_string(edits).unwrap_or_default());
        if let Some(s) = self.cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(Arc::clone(s));
        }
        let mut options = Options::default().with_tolerance(tolerance).with_edits(edits.to_vec()).with_mass(mass).with_failures_allowed(true);
        if let Some(b) = body {
            options = options.with_bodies(vec![b.into()]);
        }
        let snapshot = Arc::new(self.design.snapshot(&options).map_err(error)?);
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= MAX_CACHED {
            cache.clear();
        }
        cache.insert(key, Arc::clone(&snapshot));
        Ok(snapshot)
    }
}

/// A shape's geometry.
#[derive(Debug, Clone)]
pub(crate) enum Source {
    /// Body `body` (SolidCraft's name for it) of a design with `edits` applied.
    Design { design: Arc<DesignSource>, edits: Arc<Vec<Edit>>, body: String },
    /// A shape built in code: its mesh at every tolerance.
    Fixed(Arc<Fixed>),
}

/// A shape built in code.
#[derive(Debug)]
pub(crate) struct Fixed {
    pub(crate) mesh: SurfaceMesh,
    pub(crate) surfaces: Vec<Option<FaceSurface>>,
    pub(crate) measure: Measure,
    pub(crate) size: f64,
}

fn body_of(snapshot: &Snapshot, name: &str) -> Result<Body> {
    snapshot.body(name).cloned().ok_or_else(|| Error::Failed(format!("the design has no body `{name}`")))
}

pub(crate) fn surface(s: &solidcraft_exchange::Surface) -> Option<FaceSurface> {
    use solidcraft_exchange::Surface as S;
    Some(match s {
        S::Plane { point, normal } => FaceSurface::Plane { normal: normal.get(), point: point.0 },
        S::Cylinder { origin, axis, radius } => FaceSurface::Cylinder { origin: origin.0, axis: axis.get(), radius: radius.0 },
        S::Cone { apex, axis, half_angle } => FaceSurface::Cone { apex: apex.0, axis: axis.get(), half_angle: half_angle.0 },
        S::Sphere { center, radius } => FaceSurface::Sphere { center: center.0, radius: radius.0 },
        S::Torus { center, axis, major, minor } => FaceSurface::Torus { center: center.0, axis: axis.get(), major: major.0, minor: minor.0 },
        _ => return None,
    })
}

/// A snapshot body as the crate's mesh: its triangles tagged by face, its edges as polylines.
pub(crate) fn mesh(b: &Body) -> SurfaceMesh {
    let mut triangles = Vec::new();
    let mut tri_face = Vec::new();
    for f in &b.faces {
        for t in &f.triangles {
            triangles.push(t.map(|v| v.0));
            tri_face.push(f.index.0);
        }
    }
    SurfaceMesh {
        positions: b.positions.iter().map(|p| p.0).collect(),
        triangles,
        tri_face,
        weld_distance: 2.0 * LINEAR_TOLERANCE,
        edges: b.edges.iter().map(|e| e.path.iter().filter_map(|v| b.positions.get(v.index()).map(|p| p.0)).collect()).collect(),
        edge_faces: b.edges.iter().map(|e| e.bounds.faces().map(|f| f.0).collect()).collect(),
    }
}

impl Source {
    fn snapshot_body(&self, tolerance: Tolerance, mass: bool) -> Result<Option<Body>> {
        match self {
            Source::Design { design, edits, body } => Ok(Some(body_of(&*design.snapshot(edits, Some(body), tolerance, mass)?, body)?)),
            Source::Fixed(_) => Ok(None),
        }
    }

    /// The body at a relative tolerance, for its faces, size and shells: from the snapshot of
    /// every body the design was opened with, so opening and asking share one.
    pub(crate) fn info(&self) -> Result<Option<Body>> {
        match self {
            Source::Design { design, edits, body } => {
                Ok(Some(body_of(&*design.snapshot(edits, None, Tolerance::Relative(INFO_TOLERANCE), false)?, body)?))
            }
            Source::Fixed(_) => Ok(None),
        }
    }

    /// The boundary within `tolerance` (mm), tagged by B-rep face.
    pub(crate) fn tessellate(&self, tolerance: f64) -> Result<SurfaceMesh> {
        let tolerance = tolerance.clamp(solidcraft_exchange::MIN_TOLERANCE_MM, solidcraft_exchange::MAX_TOLERANCE_MM);
        match self {
            Source::Fixed(f) => Ok(f.mesh.clone()),
            Source::Design { .. } => Ok(self.snapshot_body(Tolerance::Absolute(tolerance), false)?.map(|b| mesh(&b)).unwrap_or_default()),
        }
    }

    pub(crate) fn surfaces(&self) -> Result<Vec<Option<FaceSurface>>> {
        match self {
            Source::Fixed(f) => Ok(f.surfaces.clone()),
            Source::Design { .. } => Ok(self.info()?.map(|b| b.faces.iter().map(|f| surface(&f.surface)).collect()).unwrap_or_default()),
        }
    }

    pub(crate) fn size(&self) -> Result<f64> {
        match self {
            Source::Fixed(f) => Ok(f.size),
            Source::Design { .. } => Ok(self.info()?.map_or(0.0, |b| b.size.0)),
        }
    }

    pub(crate) fn shells(&self) -> Result<usize> {
        match self {
            Source::Fixed(_) => Ok(1),
            Source::Design { .. } => Ok(self.info()?.map_or(0, |b| b.shells as usize)),
        }
    }

    pub(crate) fn measure(&self) -> Result<Measure> {
        match self {
            Source::Fixed(f) => Ok(f.measure),
            Source::Design { .. } => {
                let b = self.snapshot_body(Tolerance::Relative(INFO_TOLERANCE), true)?.ok_or_else(|| Error::Failed("no body".into()))?;
                let m = b.mass.ok_or_else(|| Error::Failed(format!("body `{}` was not measured", b.name)))?;
                Ok(Measure { volume: m.volume, area: m.area, centroid: m.centroid.0, bbox_min: m.min.0, bbox_max: m.max.0, faces: b.faces.len() })
            }
        }
    }
}

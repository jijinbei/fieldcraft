//! The Eulerian solid's deep interior on the GPU: [`GpuSolid`] steps a
//! [`fieldcraft_eulerian::Solid`] (mapped boundary) with the nodes far from the surface on the
//! device (`interior.wgsl`: the compact edge stress, the momentum of full control volumes and
//! the upwind advection of ζ) and everything near the surface on the CPU (the band, the cut
//! cells, the surface chains and the faces next to them), exchanging each step only the
//! values of the nodes within two rings of the seam. The two run at the same time: the device
//! steps its nodes while the CPU steps its own.
//!
//! Single precision on the device, with the stress formed from small quantities (see
//! `interior.wgsl`); the CPU in double precision is the reference, and the tests state the
//! agreement (`tests/agreement.rs`). The device is the one `fieldcraft_gpu`'s adapter policy
//! picks (`FIELDCRAFT_GPU`). A research prototype for the parallelism study of
//! docs/research/reference-map.md, not yet a solve backend.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{GpuSolid, Options, Timings};

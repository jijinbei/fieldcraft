//! The explicit step on the thread pool gives the same numbers bit for bit on any number of
//! threads (fixed partitions, sums in the sequential order).
#![allow(clippy::needless_range_loop)]

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, Settings, Shape, Solid};

/// The bits of ζ, v and the stress at every node, and of the surface stress, after `steps`
/// steps of a pressurised steel ring (the Lamé cylinder on a coarse grid: cut cells, merged
/// control volumes and several surface chains) on `threads` threads.
fn run(boundary: Boundary, threads: usize, steps: usize) -> Vec<u64> {
    fieldcraft_parallel::with_threads(threads, || {
        let h = 0.5;
        let n = (50.0 / h) as usize + 1;
        let o = -0.5 * (n as f64 - 1.0) * h;
        let g = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
        let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
        let settings = Settings { damping: 0.5 * m.p_wave_speed() / 20.0, boundary, ..Settings::default() };
        let mut s = Solid::new(g, m, Shape::Annulus { center: [0.03, -0.02], inner: 10.0, outer: 20.0 }, settings, Limits::default()).unwrap();
        s.set_ambient_field(|x| if x[0].hypot(x[1]) < 15.0 { [-100.0, -100.0, 0.0] } else { [0.0; 3] }).unwrap();
        s.set_body_force(|x| [1e-3 * x[1], -2e-3 * x[0]]).unwrap();
        s.run(steps).unwrap();
        let mut bits = Vec::new();
        for k in 0..s.grid().len() {
            for x in s.reference(k).into_iter().chain(s.velocity(k)).flatten() {
                bits.push(x.to_bits());
            }
            for x in s.stress(k).into_iter().flatten() {
                bits.push(x.to_bits());
            }
        }
        for p in s.surface_stress() {
            bits.extend(p.stress.iter().chain(p.position.iter()).map(|x| x.to_bits()));
        }
        if let Some(st) = s.cut_stats() {
            assert!(st.cut_cells > 0 && st.valid_points > 0);
            bits.extend([st.cut_cells, st.valid_points, st.merged, st.sweeps, st.mapped_faces, st.fallback_points].map(|c| c as u64));
            bits.push(st.last_change.to_bits());
        }
        bits
    })
}

#[test]
fn the_mapped_step_does_not_depend_on_the_thread_count() {
    let one = run(Boundary::Mapped, 1, 60);
    assert!(one.len() > 10_000, "{}", one.len());
    for threads in [2, 7] {
        assert!(one == run(Boundary::Mapped, threads, 60), "{threads} threads");
    }
}

#[test]
fn the_blurred_step_does_not_depend_on_the_thread_count() {
    let one = run(Boundary::Blurred, 1, 60);
    for threads in [2, 7] {
        assert!(one == run(Boundary::Blurred, threads, 60), "{threads} threads");
    }
}

#[test]
fn the_ghost_filled_step_does_not_depend_on_the_thread_count() {
    let one = run(Boundary::Ghost, 1, 60);
    for threads in [2, 7] {
        assert!(one == run(Boundary::Ghost, threads, 60), "{threads} threads");
    }
}

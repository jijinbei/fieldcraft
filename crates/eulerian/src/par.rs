//! The step's loops over nodes, cells and surface points on `fieldcraft-parallel`'s pool.
//!
//! Every helper computes each index by exactly the operations the sequential loop did, on
//! its own, and returns the results in index order (maps write them in place; filters and
//! searches split the range into blocks of a fixed size, [`BLOCK`], never derived from the
//! thread count); errors are reported for the first failing
//! index, as the sequential loop would. Sums that several indices feed (a node's net force,
//! its control volume) are formed by the caller in the sequential loop's order. The results
//! are therefore the same bit for bit on any number of threads.

use crate::Error;

/// Indices per task: large enough that a task outweighs its scheduling, small enough to
/// balance a few hundred thousand nodes over many cores.
pub(crate) const BLOCK: usize = 2048;

fn ranges(n: usize) -> Vec<std::ops::Range<usize>> {
    fieldcraft_parallel::blocks(n, BLOCK).collect()
}

/// `f(i)` for `i` in `0..n`, in index order.
pub(crate) fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    fieldcraft_parallel::map_indexed(n, BLOCK, f)
}

/// `f(i)` for `i` in `0..n`, in index order; the error of the first failing index.
pub(crate) fn try_map<T: Send>(n: usize, f: impl Fn(usize) -> Result<T, Error> + Sync + Send) -> Result<Vec<T>, Error> {
    fieldcraft_parallel::try_map_indexed(n, BLOCK, f)
}

/// Items per task for lists of costly items (cut cells, surface points, the faces next to
/// the surface, band nodes): a few thousand of them must still spread over the threads.
pub(crate) const FINE: usize = 8;

/// Items per task along a surface chain (each Newton iteration maps over every chain).
pub(crate) const CHAIN: usize = 128;

/// [`map`] over the points of a surface chain.
pub(crate) fn map_chain<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    fieldcraft_parallel::map_indexed(n, CHAIN, f)
}

/// [`map`] over a list of costly items.
pub(crate) fn map_items<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    fieldcraft_parallel::map_indexed(n, FINE, f)
}

/// [`try_map`] over a list of costly items.
pub(crate) fn try_map_items<T: Send>(n: usize, f: impl Fn(usize) -> Result<T, Error> + Sync + Send) -> Result<Vec<T>, Error> {
    fieldcraft_parallel::try_map_indexed(n, FINE, f)
}

/// The indices in `0..n` where `keep(i)`, ascending.
pub(crate) fn filter(n: usize, keep: impl Fn(usize) -> bool + Sync + Send) -> Vec<usize> {
    let r = ranges(n);
    fieldcraft_parallel::map(r.len(), |b| r.get(b).cloned().unwrap_or(0..0).filter(|&i| keep(i)).collect::<Vec<usize>>())
        .into_iter()
        .flatten()
        .collect()
}

/// The first index in `0..n` where `hit(i)`.
pub(crate) fn first(n: usize, hit: impl Fn(usize) -> bool + Sync + Send) -> Option<usize> {
    let r = ranges(n);
    fieldcraft_parallel::map(r.len(), |b| r.get(b).cloned().unwrap_or(0..0).find(|&i| hit(i))).into_iter().flatten().next()
}

/// `f(i, &mut data[i])` for every item.
pub(crate) fn update<T: Send>(data: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    fieldcraft_parallel::chunks_mut(data, BLOCK, |b, chunk| {
        for (i, x) in chunk.iter_mut().enumerate() {
            f(b * BLOCK + i, x);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_keep_index_order_and_the_first_error() {
        let n = 3 * BLOCK + 17;
        let v = fieldcraft_parallel::with_threads(5, || map(n, |i| i * 2));
        assert!(v.iter().enumerate().all(|(i, x)| *x == 2 * i));
        let e = fieldcraft_parallel::with_threads(5, || try_map(n, |i| if i % 1000 == 999 { Err(Error::Invalid(format!("{i}"))) } else { Ok(i) }));
        assert_eq!(e, Err(Error::Invalid("999".into())));
        let f = fieldcraft_parallel::with_threads(3, || filter(n, |i| i % 7 == 0));
        assert_eq!(f, (0..n).filter(|i| i % 7 == 0).collect::<Vec<_>>());
        assert_eq!(fieldcraft_parallel::with_threads(4, || first(n, |i| i > 2 * BLOCK + 3 && i % 2 == 0)), Some(2 * BLOCK + 4));
        assert_eq!(first(n, |_| false), None);
        let mut d = vec![0usize; n];
        update(&mut d, |i, x| *x = i + 1);
        assert!(d.iter().enumerate().all(|(i, x)| *x == i + 1));
    }
}

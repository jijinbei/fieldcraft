//! EXPERIMENT: move every vertex of a SolidCraft exchange snapshot by a pseudo-random amount in
//! [-amplitude, amplitude] per coordinate, to see how far the mesher's result moves.
//!
//! `perturb_snapshot <in.json> <out.json> <seed> <amplitude mm>`
use serde_json::Value;

/// SplitMix64: a small, fixed generator, so a seed gives the same file everywhere.
fn next(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [input, output, seed, amplitude] = a.as_slice() else {
        return Err("usage: perturb_snapshot <in.json> <out.json> <seed> <amplitude mm>".into());
    };
    let mut state: u64 = seed.parse().map_err(|e| format!("seed: {e}"))?;
    let amplitude: f64 = amplitude.parse().map_err(|e| format!("amplitude: {e}"))?;
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let mut snapshot: Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
    for body in snapshot["bodies"].as_array_mut().into_iter().flatten() {
        for p in body["positions"].as_array_mut().into_iter().flatten() {
            for c in p.as_array_mut().into_iter().flatten() {
                let x = c.as_f64().ok_or("a position that is not a number")?;
                *c = serde_json::json!(x + amplitude * (2.0 * next(&mut state) - 1.0));
            }
        }
    }
    std::fs::write(output, snapshot.to_string()).map_err(|e| format!("{output}: {e}"))
}

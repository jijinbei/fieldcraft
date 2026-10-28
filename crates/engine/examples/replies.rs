//! EXPERIMENT: run a script and print each command's reply (one JSON line each).
fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: replies <script.json>")?;
    let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let steps: Vec<serde_json::Value> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut s = fieldcraft_engine::Session::default();
    for (k, step) in steps.iter().enumerate() {
        let command = step["command"].as_str().unwrap_or_default();
        let t = std::time::Instant::now();
        match s.execute(command, &step["params"]) {
            Ok(r) => println!("{k} {command} {:.2}s {}", t.elapsed().as_secs_f64(), r),
            Err(e) => return Err(format!("step {k} ({command}): {e}")),
        }
    }
    Ok(())
}

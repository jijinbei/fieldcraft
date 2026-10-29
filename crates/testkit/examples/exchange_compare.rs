//! EXPERIMENT: run oracle cases twice, opening `design.json` as today and importing a SolidCraft
//! exchange snapshot of the same design instead, and print each quantity both ways.
//!
//! `exchange_compare <snapshots dir> <case>...`: the snapshot of case `c` is `<dir>/<c>.json`.
//! In the snapshot run, `mesh.*` steps that name a design body (`"body": "Bracket"`) name the
//! imported surface of that body instead (`"surface": 0`).
use std::path::Path;

use fieldcraft_engine::Session;
use fieldcraft_testkit::oracle::{Expected, run_case, run_recipe};
use serde_json::{Value, json};

fn read(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The recipe with design bodies in `mesh.*` steps replaced by the imported surfaces of the same name.
fn on_surfaces(recipe: &Value, surfaces: &[(String, u64)]) -> Value {
    let mut recipe = recipe.clone();
    for step in recipe.as_array_mut().into_iter().flatten() {
        let mesh = step.get("command").and_then(Value::as_str).is_some_and(|c| c.starts_with("mesh."));
        let Some(params) = step.get_mut("params").and_then(Value::as_object_mut) else { continue };
        if !mesh {
            continue;
        }
        let body = params.get("body").and_then(Value::as_str).map(str::to_string);
        if let Some((_, k)) = body.and_then(|b| surfaces.iter().find(|(n, _)| *n == b)) {
            params.remove("body");
            params.insert("surface".into(), json!(k));
        }
    }
    recipe
}

fn snapshot_run(dir: &Path, snapshot: &Path, expected: &Expected) -> Result<Vec<Result<f64, String>>, String> {
    let mut session = Session::default();
    let reply = session.execute("geometry.import.mesh", &json!({"path": snapshot})).map_err(|e| format!("import: {e}"))?;
    let surfaces: Vec<(String, u64)> =
        reply["surfaces"].as_array().into_iter().flatten().filter_map(|s| Some((s["name"].as_str()?.to_string(), s["index"].as_u64()?))).collect();
    let recipe = on_surfaces(&read(&dir.join("recipe.json"))?, &surfaces);
    run_recipe(&mut session, &recipe, expected)
}

fn show(r: &Result<f64, String>) -> String {
    match r {
        Ok(x) => format!("{x:.6e}"),
        Err(e) => format!("error: {e}"),
    }
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let snapshots = args.next().ok_or("usage: exchange_compare <snapshots dir> <case>...")?;
    for case in args {
        let dir = Path::new("tests/oracle").join(&case);
        let expected = Expected::parse(&read(&dir.join("expected.json"))?).map_err(|e| format!("{case}: {e}"))?;
        let native = run_case(&dir, &expected);
        let snapshot = snapshot_run(&dir, &Path::new(&snapshots).join(format!("{case}.json")), &expected);
        println!("== {case}");
        let rows = |r: &Result<Vec<Result<f64, String>>, String>| match r {
            Ok(v) => v.clone(),
            Err(e) => vec![Err(e.clone()); expected.quantities.len()],
        };
        for ((q, a), b) in expected.quantities.iter().zip(rows(&native)).zip(rows(&snapshot)) {
            let verdict = |r: &Result<f64, String>| match (r, q.reference) {
                (Ok(x), Some(reference)) => {
                    if q.tolerance.allows(reference, *x) {
                        "pass"
                    } else {
                        "FAIL"
                    }
                }
                (Ok(_), None) => "-",
                (Err(_), _) => "ERROR",
            };
            let diff = match (&a, &b) {
                (Ok(x), Ok(y)) if *x != 0.0 => format!("{:+.2e}", (y - x) / x.abs()),
                _ => "-".into(),
            };
            println!(
                "{}\treference {}\tnative {} {}\tsnapshot {} {}\trelative difference {diff}",
                q.id,
                q.reference.map_or("-".into(), |r| format!("{r:.6e}")),
                show(&a),
                verdict(&a),
                show(&b),
                verdict(&b),
            );
        }
    }
    Ok(())
}

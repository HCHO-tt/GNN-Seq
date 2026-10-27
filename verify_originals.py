"""Check existing code and the frozen Seq snapshot against pre-work hashes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "original_files.sha256.json").read_text(encoding="utf-8"))
failures = []
for relative, expected in manifest.items():
    path = root.parent / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(relative)
    if relative.startswith("Seq-MADAC-main\\"):
        snapshot = root / "vendor/Seq-MADAC" / Path(relative).relative_to("Seq-MADAC-main")
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
            failures.append(f"snapshot:{relative}")
if failures:
    raise SystemExit("Changed/missing original files:\n" + "\n".join(failures))
print(f"PASS: {len(manifest)} pre-existing source/config/document files unchanged; Seq snapshot matches.")

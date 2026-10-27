"""Import a frozen local Seq snapshot; never import sibling experiment code."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "vendor" / "Seq-MADAC"
for directory in (SEQ, SEQ / "algos" / "madac"):
    sys.path.insert(0, str(directory))

# Upstream learners/__init__.py references unavailable HAPPO classes.
# Loading this file directly preserves SADNLearner byte-for-byte.
spec = importlib.util.spec_from_file_location(
    "_original_sadn", SEQ / "algos/madac/learners/sadn_learner.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SADNLearner = module.SADNLearner

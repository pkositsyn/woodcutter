"""Regenerate golden files from the vendored reference (pulp) solver.

Run from the repository root:
    python3 reference/gen_golden.py
Requires python3 + pulp. Padding is fixed at 5 to match cutting.Options.
"""
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from cutting import parse_line, min_material_cutting  # noqa: E402

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "internal", "cutting", "testdata")


def load(path):
    stock_lines, req_lines, section, seen = [], [], "stock", False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "---":
                section, seen = "req", True
                continue
            (stock_lines if section == "stock" else req_lines).append(line)
    stock = [parse_line(l) for l in stock_lines]
    counter = Counter()
    for length, count in (parse_line(l) for l in req_lines):
        counter[length] += count
    return stock, list(counter.items())


def main():
    for path in sorted(glob.glob(os.path.join(TESTDATA, "*.txt"))):
        stock, requirements = load(path)
        total, _ = min_material_cutting(requirements, stock, padding=5, verbose=False)
        golden = os.path.splitext(path)[0] + ".golden.json"
        with open(golden, "w", encoding="utf-8") as fh:
            json.dump({"total_material": total, "feasible": total != -1}, fh)
        print(f"{os.path.basename(path)} -> total_material={total}")


if __name__ == "__main__":
    main()

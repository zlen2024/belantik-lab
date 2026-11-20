"""Check Lean graph/color literals against the Python THRML audit graph."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_thrml_four_spin_joint import BLOCKS, EDGES

source = (ROOT / 'theory/WordEnergy/FourSpinGraph.lean').read_text(encoding='utf-8-sig')
edge_section = source.split('def edges :', 1)[1].split('-- Exact blocks', 1)[0]
edges = [tuple(map(int, p)) for p in re.findall(r'\((\d+),(\d+)\)', edge_section)]
color_section = source.split('def color :', 1)[1].split('theorem schedule_valid', 1)[0]
colors = {int(k): int(v) for k, v in re.findall(r'\| (\d+) => (\d+)', color_section)}
expected_colors = {node: block for block, nodes in enumerate(BLOCKS) for node in nodes}
if edges != EDGES or colors != expected_colors:
    raise AssertionError('Lean graph and THRML graph disagree')
print('PASS: 26 ordered edges and all 12 node colors match the THRML audit graph')

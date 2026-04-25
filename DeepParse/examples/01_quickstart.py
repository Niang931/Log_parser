"""DeepParse quickstart — three lines from paper Listing 1.

Run from the repo root:
    python examples/01_quickstart.py
"""
from deepparse import Drain, synth_masks

sys_logs = [
    "2024-01-15 10:30:45 INFO User john_doe logged in from IP 192.168.1.100",
    "2024-01-15 10:30:46 INFO User alice    logged in from IP 192.168.1.101",
    "2024-01-15 10:30:47 INFO User bob      logged in from IP 192.168.1.102",
    "2024-01-15 10:30:48 WARN Memory usage at 85 percent for host 10.0.0.5",
    "2024-01-15 10:30:49 ERROR Database timeout after 45 seconds",
]

# Step 1 — synthesise a regex bundle from a small sample (offline = no LLM).
patterns = synth_masks(sys_logs, sample_size=50, temperature=0, max_length=512)
print(patterns)
print(f"synthesised {len(patterns)} masks: {[p['label'] for p in patterns]}")

# Step 2 — load the bundle into Drain and parse every line.
drain = Drain()
drain.load_masks(patterns)
parsed = drain.parse_all(sys_logs)

print(f"\n{len(parsed)} templates produced:")
for original, template in zip(sys_logs, parsed):
    print(f"  {original}")
    print(f"  -> {template}\n")

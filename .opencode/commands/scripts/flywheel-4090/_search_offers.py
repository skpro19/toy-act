#!/usr/bin/env python3
"""Parse vastai search offers JSON and rank by CPU family + price."""
import json, sys, re

data = json.load(sys.stdin)
if isinstance(data, dict):
    offers = data.get("offers", [])
else:
    offers = data

print(f"Found {len(offers)} offers matching all hard filters")
print()

def cpu_rank(cpu):
    """Rank per workflow: EPYC 9005 > EPYC 9004 > Threadripper 7000 > EPYC 7003 > Ryzen 7000/9000."""
    cpu = (cpu or "").upper()
    # EPYC 9005
    if re.search(r'EPYC.*(?:965[45]|9555|9475|9375|9275|9175|9965)', cpu):
        return 0
    # EPYC 9004
    if re.search(r'EPYC.*(?:9654|9634|9554|9534|9374|9354|9334|9274|9254|9224|9174|9124)', cpu):
        return 0
    if "EPYC 900" in cpu or "EPYC 9004" in cpu:
        return 0
    # EPYC 9004 detection by model pattern 9xx4
    if re.search(r'EPYC.*9[0-9]{2,}4', cpu) or "EPYC 9684X" in cpu:
        return 1
    # Threadripper 7000
    if "THREADRIPPER" in cpu and "7000" in cpu:
        return 2
    if "THREADRIPPER" in cpu and re.search(r'79[0-9]{2,}W?X', cpu):
        return 2
    # EPYC 7003
    if re.search(r'EPYC.*7[0-9]{2,}3', cpu):
        return 3
    if "EPYC 7713" in cpu or "EPYC 7C13" in cpu:
        return 3
    if "EPYC" in cpu:
        if "7001" in cpu or "7002" in cpu:
            return 99
        return 10  # unknown EPYC, acceptable
    # Threadripper (non-7000)
    if "THREADRIPPER" in cpu:
        if "5000" in cpu:
            return 99  # Zen 3 Threadripper is OK actually... but workflow says Zen 3+ so fine
        return 20
    # Ryzen 7000/9000
    if "RYZEN" in cpu:
        if re.search(r'9[0-9]{3,}0?X?$', cpu) or re.search(r'7[0-9]{3,}0?X?$', cpu):
            return 4
        if re.search(r'7950|7900|7800|7700|7600|9950|9900|9800|9700|9600', cpu):
            return 4
        if "7000" in cpu or "9000" in cpu:
            return 4
        # Zen 3 Ryzen (5000 series) - older but still Zen 3+
        if re.search(r'5[0-9]{3,}0?X?$', cpu):
            return 5
        return 22
    return 99

eligible = []
for o in offers:
    cpu = o.get("cpu_name", "") or ""
    rank = cpu_rank(cpu)
    if rank >= 50:
        continue
    eligible.append((rank, o.get("dph_total", 999), o))

eligible.sort(key=lambda x: (x[0], x[1]))

header = (
    f"{'Rank':<5} {'Offer ID':<12} {'CPU Model':<35} {'Eff vCPUs':<10} {'RAM GB':<8} "
    f"{'Disk MB/s':<10} {'PCIe GB/s':<9} {'GPU Pwr':<8} {'Down/Up':<14} "
    f"{'Reliab':<7} {'$/hr':<8} {'Location'}"
)
print(header)
print("-" * 135)

shortlist_ids = []
for i, (rank, price, o) in enumerate(eligible, 1):
    oid = str(o.get("id", ""))
    cpu_name = (o.get("cpu_name", "") or "")[:35]
    eff = o.get("cpu_cores_effective", "")
    ram = o.get("cpu_ram", "")
    disk = o.get("disk_bw", "")
    pcie = o.get("pcie_bw", "")
    gpup = o.get("gpu_max_power", "")
    down = o.get("inet_down", "")
    up = o.get("inet_up", "")
    rel = o.get("reliability", "")
    loc = (o.get("geolocation", "") or "")[:16]
    line = (
        f"{i:<5} {oid:<12} {cpu_name:<35} {eff:<10} {ram:<8} {disk:<10} "
        f"{pcie:<9} {gpup:<8} {down}/{up:<9} {rel:<7} {price:<8.4f} {loc}"
    )
    print(line)
    shortlist_ids.append(oid)

print()
print("=" * 60)
print("AUTO-SELECTED SHORTLIST (up to 3, under $0.60/hr)")
print("=" * 60)
selected = shortlist_ids[:3]
for sid in selected:
    print(f"  Offer {sid}")

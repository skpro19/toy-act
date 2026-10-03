#!/usr/bin/env bash
# Structured probe. Missing cgroup information is an error, not unlimited capacity.
set -euo pipefail
export LC_ALL=C
/opt/conda/bin/python - <<'PY'
import json
import os
from pathlib import Path
import subprocess


def output(args) -> str:
    return subprocess.check_output(args, text=True, timeout=30).strip()


def cgroup_path(*, filename: str, controller: str) -> Path:
    records = Path('/proc/self/cgroup').read_text().splitlines()
    for record in records:
        _, controllers, relative = record.split(':', 2)
        if controllers == '' or controller in controllers.split(','):
            root = Path('/sys/fs/cgroup') if controllers == '' else Path('/sys/fs/cgroup') / controller
            candidates = [root / relative.lstrip('/') / filename, root / filename]
            for path in candidates:
                if path.is_file():
                    return path
    raise RuntimeError(f'Cannot locate cgroup limit {filename}')


allowed = os.sched_getaffinity(0)
cores = set()
for line in output(['lscpu', '-p=CPU,CORE,SOCKET,ONLINE']).splitlines():
    if line.startswith('#'):
        continue
    cpu, core, socket, online = line.split(',')
    if int(cpu) in allowed and online == 'Y':
        cores.add((int(core), int(socket)))
model = next(line.split(':', 1)[1].strip() for line in output(['lscpu']).splitlines()
             if line.startswith('Model name:'))
v2 = any(line.startswith('0::') for line in Path('/proc/self/cgroup').read_text().splitlines())
if v2:
    quota, period = cgroup_path(filename='cpu.max', controller='cpu').read_text().split()
    cpu_quota = None if quota == 'max' else int(quota) / int(period)
    memory = cgroup_path(filename='memory.max', controller='memory').read_text().strip()
else:
    quota = int(cgroup_path(filename='cpu.cfs_quota_us', controller='cpu').read_text())
    period = int(cgroup_path(filename='cpu.cfs_period_us', controller='cpu').read_text())
    cpu_quota = None if quota == -1 else quota / period
    memory = cgroup_path(filename='memory.limit_in_bytes', controller='memory').read_text().strip()
visible_memory = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                      if line.startswith('MemTotal:'))
memory_bytes = visible_memory if memory == 'max' or int(memory) >= 2**62 else min(int(memory), visible_memory)
query = ('name,memory.total,power.limit,pcie.link.gen.max,pcie.link.width.max,'
         'clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown')
gpus = []
for line in output(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits']).splitlines():
    name, memory, power, gen, width, thermal, brake = [field.strip() for field in line.split(',')]
    gpus.append(dict(name=name, memory_mib=float(memory), power_watts=float(power), pcie_gen=int(gen),
                     pcie_width=int(width), thermal_slowdown=thermal, power_brake_slowdown=brake))
disk = os.statvfs('/workspace')
print(json.dumps(dict(cpu_model=model, physical_cores=len(cores), logical_cpus=len(allowed),
                      cpu_quota=cpu_quota, memory_bytes=memory_bytes,
                      workspace_available_bytes=disk.f_bavail * disk.f_frsize, gpus=gpus)))
PY

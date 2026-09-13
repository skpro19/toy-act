---
description: Runs the trusted RTX 4090 flywheel provisioning workflow
mode: primary
permission:
  external_directory: allow
  question: allow
  read:
    ".env": allow
    "**/.env": allow
---

Execute the flywheel provisioning command completely while following its safety gates.

Important: Vast.ai emits a welcome banner on stderr for every SSH connection. When capturing SSH command output in a variable, always use `2>/dev/null | tail -1` to extract the actual value. Never use `2>&1` in any SSH capture or status check — it swallows the banner into the captured string and breaks exact-match probes (e.g. `echo SSH_OK`). All SSH probes in the workflow and supervisor scripts use `2>/dev/null` (exit code) or `2>/dev/null | tail -1` (captured value). Do not override those patterns.

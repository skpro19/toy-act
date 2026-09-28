---
description: List all currently running Vast.ai instances in a table
agent: build
---

Using the Vast.ai REST API (not the CLI), list every instance whose
`actual_status` is `running`, in tabular form.

Fetch the instances and reduce them to tab-separated columns with `jq`:

```bash
source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/instances" \
| jq -r '
  ["ID","Label","GPU","GPU %","CPU %","RAM (GB)","Disk (GB)","$/hr","Uptime (h)","SSH"],
  (.instances[] | select(.actual_status == "running") |
    [ (.id | tostring),
      .label,
      "\(.gpu_name // "?") x\(.num_gpus)",
      "\((.gpu_util // 0) | floor)%",
      "\((.cpu_util // 0) | floor)%",
      "\((.mem_usage // 0) | round)/\((.mem_limit // 0) | round)",
      "\((.disk_usage // 0) | round)",
      "\(((.dph_total // 0) * 100 | round) / 100)",
      "\(((.uptime_mins // 0) / 6 | round) / 10)",
      "\(.ssh_host):\(.ssh_port)" ])
  | @tsv
'
```

Render the output as a single GitHub-flavored markdown table and state how many
instances are running. If no instances are running, show "No instances running."

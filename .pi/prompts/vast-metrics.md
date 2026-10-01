---
description: Show all Vast.ai instances (with uptime) and the current account balance in one table
---

Using the Vast.ai REST API (not the CLI), fetch and display every instance and
the account balance in a single response.

1. **Instances** — `GET /api/v0/instances`. For each entry in `.instances`,
   report its ID, label, GPU, status (`actual_status`), how long it has been
   running, and its cost per hour.
2. **Balance** — the `credit` field from `GET /api/v0/users/current`, shown as
   USD.

Use `source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY"` for the
API calls and `jq` to reduce the instance list to tab-separated columns:

```bash
source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/instances" \
| jq -r '
  ["ID","Label","GPU","Status","Uptime (h)","$/hr"],
  (.instances[] |
    [ (.id | tostring),
      (.label // "?"),
      "\(.gpu_name // "?") x\(.num_gpus // 0)",
      (.actual_status // "?"),
      "\(((.uptime_mins // 0) / 6 | round) / 10)",
      "\(((.dph_total // 0) * 100 | round) / 100)" ])
  | @tsv
'

source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/users/current" \
| jq -r '.credit'
```

Render the instances as a single GitHub-flavored markdown table, then report the
balance on its own line as a single dollar amount, e.g. `Balance: $4.98`.
Uptime is shown in hours to one decimal place. If no instances exist, show
"No instances found."

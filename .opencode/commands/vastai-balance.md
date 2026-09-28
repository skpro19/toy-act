---
description: Show the current Vast.ai account balance in USD
agent: build
---

Using the Vast.ai REST API (not the CLI), fetch the current account balance and
display it as USD.

Use the `credit` field from `GET /api/v0/users/current`:

```bash
source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/users/current" \
| jq -r '.credit'
```

Report the balance as a single dollar amount, e.g. `$19.91`.

---
description: Show Vast.ai instance and balance metrics
agent: build
---

Using the Vast.ai REST API (not the CLI), fetch and display the following metrics in a single response:

1. **Number of running instances** — count from `GET /api/v0/instances`
2. **Current balance** — `credit` field from `GET /api/v0/users/current`, displayed as USD

Use `source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY"` for API calls and `jq` for JSON processing. If no instances are running, show "No instances running."

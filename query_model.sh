#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="http://localhost:1234/v1/chat/completions"
API_KEY="none"

response=$(curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "what is love? In 2 sentences."
      }
    ],
    "temperature": 0.0
  }')

if command -v jq >/dev/null 2>&1; then
  echo "$response" | jq -r '.choices[0].message.content // .'
else
  echo "$response"
fi

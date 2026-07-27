#!/usr/bin/env bash
# macOS / Linux wrapper. All logic lives in local.py.
exec python3 "$(dirname "$0")/local.py" "$@"

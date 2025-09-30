# File: docker/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail
# Minimal entrypoint; lets us pass through any CLI
exec "$@
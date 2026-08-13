#!/usr/bin/env bash
set -Eeuo pipefail

service_name="${BNHMM_SERVICE_NAME:-bnhmm-web.service}"
host="${BNHMM_WEB_HOST:-127.0.0.1}"
port="${BNHMM_WEB_PORT:-8765}"
api_root="http://${host}:${port}/api"
force=0

case "${1:-}" in
    "") ;;
    --force) force=1 ;;
    -h|--help)
        echo "Usage: ./stop.sh [--force]"
        echo "Without --force, shutdown is refused while training jobs are active."
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        echo "Usage: ./stop.sh [--force]" >&2
        exit 2
        ;;
esac

if ! systemctl --user is-active --quiet "${service_name}"; then
    echo "BNHMM is already stopped."
    exit 0
fi

config_json=""
active_jobs=""
if config_json=$(curl -fsS "${api_root}/config" 2>/dev/null); then
    active_jobs=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["active_jobs"])' <<<"${config_json}")
fi

if [[ -z "${active_jobs}" && ${force} -eq 0 ]]; then
    echo "Cannot verify active jobs; use ./stop.sh --force to stop anyway." >&2
    exit 3
fi

if [[ "${active_jobs:-0}" -gt 0 && ${force} -eq 0 ]]; then
    echo "Refusing to stop: ${active_jobs} training job(s) are active." >&2
    echo "Wait for completion or run ./stop.sh --force." >&2
    exit 4
fi

if [[ "${active_jobs:-0}" -gt 0 && ${force} -eq 1 ]]; then
    while IFS= read -r identifier; do
        [[ -z "${identifier}" ]] && continue
        curl -fsS -X POST "${api_root}/jobs/${identifier}/cancel" >/dev/null || true
        echo "Cancelled job: ${identifier}"
    done < <(
        curl -fsS "${api_root}/jobs" 2>/dev/null |
            python3 -c 'import json, sys; print("\n".join(item["id"] for item in json.load(sys.stdin) if item["status"] in {"queued", "running"}))'
    )
fi

systemctl --user stop "${service_name}"
if systemctl --user is-active --quiet "${service_name}"; then
    echo "BNHMM did not stop cleanly." >&2
    exit 1
fi

echo "BNHMM stopped."

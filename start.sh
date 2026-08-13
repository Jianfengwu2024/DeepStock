#!/usr/bin/env bash
set -Eeuo pipefail

service_name="${BNHMM_SERVICE_NAME:-bnhmm-web.service}"
host="${BNHMM_WEB_HOST:-127.0.0.1}"
port="${BNHMM_WEB_PORT:-8765}"
health_url="http://${host}:${port}/health"

if ! systemctl --user cat "${service_name}" >/dev/null 2>&1; then
    echo "Service ${service_name} is not installed." >&2
    echo "Install deploy/bnhmm-web.service in ~/.config/systemd/user first." >&2
    exit 1
fi

systemctl --user start "${service_name}"

for _ in $(seq 1 30); do
    if curl -fsS "${health_url}" >/dev/null 2>&1; then
        public_host="${BNHMM_PUBLIC_HOST:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
        public_host="${public_host:-127.0.0.1}"
        echo "BNHMM started: http://${public_host}:${port}"
        exit 0
    fi
    if ! systemctl --user is-active --quiet "${service_name}"; then
        echo "BNHMM failed to start. Recent service log:" >&2
        journalctl --user -u "${service_name}" -n 30 --no-pager >&2
        exit 1
    fi
    sleep 1
done

echo "BNHMM did not become healthy within 30 seconds." >&2
journalctl --user -u "${service_name}" -n 30 --no-pager >&2
exit 1

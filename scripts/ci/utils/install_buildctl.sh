#!/bin/bash
# Install buildctl from a verified archive, shared safely across fresh runners.
set -euo pipefail

archive_url=${1:?archive URL is required}
expected_sha=${2:?archive SHA256 is required}
cache_dir=${3:?cache directory is required}
binary_path=${4:?destination binary is required}
archive="$cache_dir/buildkit.tar.gz"
partial="$archive.part"

mkdir -p "$cache_dir" "$(dirname "$binary_path")"
exec 9>"$cache_dir/.lock"
# Cover the bounded download budget (8 * 240s plus backoff) on a cold cache.
flock -w 2100 9 || { echo "::error::timed out waiting for the buildctl cache lock"; exit 1; }

valid_archive() {
    [ -f "$1" ] && printf '%s  %s\n' "$expected_sha" "$1" | sha256sum --check --status
}

if ! valid_archive "$archive"; then
    rm -f "$archive"
    for attempt in $(seq 1 8); do
        echo "Downloading buildctl archive (attempt $attempt/8)"
        curl_status=0
        http_status=$(curl -fsSL --connect-timeout 20 --max-time 240 \
            --continue-at - --write-out '%{http_code}' -o "$partial" "$archive_url") || curl_status=$?
        if valid_archive "$partial"; then
            mv "$partial" "$archive"
            break
        fi
        # Keep partial data after an interrupted transfer. Restart when the
        # server cannot resume, the offset is invalid, or a complete file is bad.
        if [ "$curl_status" -eq 33 ] || [ "$http_status" = 416 ] || [ "$curl_status" -eq 0 ]; then
            rm -f "$partial"
        fi
        echo "::warning::buildctl download attempt $attempt failed verification (curl=$curl_status, HTTP=$http_status)"
        if [ "$attempt" -lt 8 ]; then sleep $((attempt * 3)); fi
    done
fi

valid_archive "$archive" || { echo "::error::could not download a verified buildctl archive"; exit 1; }
# Extract only the client into a private, temporary path. Never execute a
# cached binary without validating its source archive, or publish a partial file.
temporary_binary=$(mktemp "${binary_path}.XXXXXX")
trap 'rm -f "$temporary_binary"' EXIT
tar -xzOf "$archive" bin/buildctl > "$temporary_binary"
chmod 0755 "$temporary_binary"
"$temporary_binary" --version
mv -f "$temporary_binary" "$binary_path"

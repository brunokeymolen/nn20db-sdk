#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/install-sdk.sh <linux|esp32> <tarball-path-or-url>

Examples:
  ./scripts/install-sdk.sh linux /tmp/nn20db-sdk-linux-v0.1.0.tar.gz
  ./scripts/install-sdk.sh esp32 https://github.com/OWNER/REPO/releases/download/v0.1.0/nn20db-sdk-esp32-v0.1.0.tar.gz
EOF
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

target="$1"
source_ref="$2"

case "$target" in
    linux|esp32) ;;
    *)
        echo "error: target must be 'linux' or 'esp32'" >&2
        exit 2
        ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
sdk_root="${repo_root}/sdk/${target}"
tmp_dir="$(mktemp -d)"
archive_path="${tmp_dir}/payload.tar.gz"
listing_path="${tmp_dir}/listing.txt"

cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

if [[ "${source_ref}" =~ ^https?:// ]]; then
    echo "Downloading ${source_ref}"
    curl -fL --retry 3 --retry-delay 1 -o "${archive_path}" "${source_ref}"
else
    if [[ ! -f "${source_ref}" ]]; then
        echo "error: tarball not found: ${source_ref}" >&2
        exit 1
    fi
    cp "${source_ref}" "${archive_path}"
fi

tar -tzf "${archive_path}" > "${listing_path}"
top_level="$(head -n1 "${listing_path}" | cut -d/ -f1)"
if [[ -z "${top_level}" ]]; then
    echo "error: could not determine tarball root directory" >&2
    exit 1
fi

mkdir -p "${sdk_root}"
rm -rf "${sdk_root:?}/${top_level}"
tar -xzf "${archive_path}" -C "${sdk_root}"

ln -sfn "${top_level}" "${sdk_root}/current"

echo "Installed ${target} SDK:"
echo "  ${sdk_root}/current"

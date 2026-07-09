#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
install_aiter=0

usage() {
  cat <<'EOF'
usage: scripts/benchmark/prepare_sources.sh [--install-aiter]

Clone the third-party source checkouts required by the artifact benchmark
backends. The source directories are ignored by git; the pinned revisions live in
third_party/*.source.
EOF
}

while (($#)); do
  case "$1" in
    --install-aiter)
      install_aiter=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

source_value() {
  local file="$1"
  local key="$2"
  awk -F= -v key="$key" '$1 == key { print $2 }' "$file"
}

checkout_source() {
  local name="$1"
  local source_file="$2"
  local dest="$3"
  local upstream ref commit checkout_ref actual submodules

  upstream="$(source_value "$source_file" upstream)"
  ref="$(source_value "$source_file" ref)"
  commit="$(source_value "$source_file" commit)"
  submodules="$(source_value "$source_file" submodules)"
  checkout_ref="${ref:-$commit}"

  if [[ -z "$upstream" || -z "$commit" ]]; then
    echo "invalid source lock: $source_file" >&2
    exit 1
  fi

  if [[ -e "$dest" && ! -d "$dest/.git" ]]; then
    echo "$dest exists but is not a git checkout" >&2
    exit 1
  fi

  if [[ ! -d "$dest/.git" ]]; then
    git clone "$upstream" "$dest"
  fi

  if [[ -n "$ref" ]]; then
    git -C "$dest" fetch --tags origin "$ref"
  else
    git -C "$dest" fetch --tags origin "+refs/heads/*:refs/remotes/origin/*"
  fi
  git -C "$dest" -c advice.detachedHead=false checkout --detach "$checkout_ref"
  if [[ "$submodules" == "recursive" ]]; then
    git -C "$dest" submodule update --init --recursive
  fi

  actual="$(git -C "$dest" rev-parse HEAD)"
  if [[ "$actual" != "$commit" ]]; then
    echo "$name checkout mismatch: expected $commit, got $actual" >&2
    exit 1
  fi
}

checkout_source \
  "AITER" \
  "$repo_root/third_party/aiter.source" \
  "$repo_root/third_party/aiter"

checkout_source \
  "HipKittens" \
  "$repo_root/third_party/HipKittens.source" \
  "$repo_root/third_party/HipKittens"

if ((install_aiter)); then
  aiter_package="$(source_value "$repo_root/third_party/aiter.source" python_package)"
  aiter_version="$(source_value "$repo_root/third_party/aiter.source" python_version)"
  if python3 - "$aiter_package" "$aiter_version" <<'PY'
import importlib.metadata as md
import sys

package, expected = sys.argv[1], sys.argv[2]
try:
    installed = md.version(package)
except md.PackageNotFoundError:
    sys.exit(1)
sys.exit(0 if installed == expected else 1)
PY
  then
    echo "$aiter_package==$aiter_version already installed"
  else
    python3 -m pip install --no-deps "$repo_root/third_party/aiter"
  fi
fi

echo "benchmark source checkouts ready"

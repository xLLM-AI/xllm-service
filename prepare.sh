#!/usr/bin/env bash
set -e

# The vendored etcd-cpp-apiv3 links against the system cpprest package on
# Debian/Ubuntu builds. Do not install packages implicitly in environments where
# apt/dpkg is unavailable; instead print an actionable message.
if command -v dpkg >/dev/null 2>&1 && ! dpkg -s libcpprest-dev >/dev/null 2>&1; then
  if [ "$(id -u)" -eq 0 ] && command -v apt-get >/dev/null 2>&1; then
    echo "libcpprest-dev not found, installing ..."
    apt-get update
    apt-get install -y libcpprest-dev
  else
    echo "libcpprest-dev is missing. Please install it first, for example:" >&2
    echo "  sudo apt-get install libcpprest-dev" >&2
    exit 1
  fi
fi

cd ./third_party/cpprestsdk
# Apply the patch only when it is not already applied, so re-running prepare.sh
# under `set -e` stays idempotent instead of aborting on an already-patched tree.
if ! git apply --reverse --check ../custom_cache/cpprestsdk.patch >/dev/null 2>&1; then
  git apply ../custom_cache/cpprestsdk.patch
fi

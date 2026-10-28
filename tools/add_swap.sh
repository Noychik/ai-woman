#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Запусти через sudo: sudo bash tools/add_swap.sh" >&2
  exit 1
fi

SWAPFILE="/swapfile"
SIZE="16G"

if swapon --show=NAME --noheadings | grep -qx "${SWAPFILE}"; then
  echo "Swap ${SWAPFILE} уже активен."
  swapon --show
  free -h
  exit 0
fi

if [[ ! -e "${SWAPFILE}" ]]; then
  echo "Создаю ${SIZE} swapfile на btrfs..."
  btrfs filesystem mkswapfile --size "${SIZE}" "${SWAPFILE}"
else
  echo "Файл ${SWAPFILE} уже существует, использую его."
  chmod 600 "${SWAPFILE}"
  if ! file "${SWAPFILE}" | grep -qi swap; then
    mkswap "${SWAPFILE}"
  fi
fi

chmod 600 "${SWAPFILE}"
swapon "${SWAPFILE}"

if ! grep -qE '^[[:space:]]*/swapfile[[:space:]]' /etc/fstab; then
  echo "/swapfile none swap defaults,pri=10 0 0" >> /etc/fstab
  echo "Добавил /swapfile в /etc/fstab."
fi

echo
echo "Готово:"
swapon --show
free -h

#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Vaibhav — final power over this project: Vaibhav.
# Launcher: enforces ~/ml_env venv only (never system pip), max threads for i5-1335U.
set -euo pipefail
# shellcheck disable=SC1091
source "$HOME/ml_env/bin/activate"
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
exec python /home/neko/coin-flip/vaibhav_market_cli.py "$@"

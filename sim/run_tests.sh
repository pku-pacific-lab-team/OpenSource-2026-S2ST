#!/usr/bin/env bash
# Copyright 2026 School of Integrated Circuits, Peking University
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# Run the unit / integration testbenches for the pe_array_group RTL.
#
# Usage:
#   sim/run_tests.sh [-s vcs|xsim|iverilog] [tb_name ...]
#
# With no tb_name, every testbench under tb/ is run. Build outputs go to
# sim/build/<simulator>/<tb_name>/. Pass FSDB=1 with VCS to dump waveform.fsdb.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM="${SIM:-vcs}"

while getopts "s:h" opt; do
  case "$opt" in
    s) SIM="$OPTARG" ;;
    h) sed -n '4,11p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))

mapfile -t RTL < <(cat "$ROOT"/rtl/*.f | grep -vE '^\s*(//|$)' | sed "s#^#$ROOT/#")

if [ "$#" -gt 0 ]; then
  TBS=("$@")
else
  mapfile -t TBS < <(find "$ROOT/tb" -name '*_tb.sv' -exec basename {} .sv \; | sort)
fi

pass=0
fail=0
failed=()

for tb in "${TBS[@]}"; do
  tb_file="$(find "$ROOT/tb" -name "${tb}.sv" | head -n 1)"
  if [ -z "$tb_file" ]; then
    echo "[SKIP] $tb (not found)"
    continue
  fi

  work="$ROOT/sim/build/$SIM/$tb"
  mkdir -p "$work"
  log="$work/sim.log"

  (
    cd "$work" || exit 1
    case "$SIM" in
      vcs)
        defs=()
        [ "${FSDB:-0}" = "1" ] && defs+=(+define+FSDB -kdb -debug_access+all)
        vcs -full64 -sverilog -timescale=1ns/1ps "${defs[@]}" \
          "${RTL[@]}" "$tb_file" -top "$tb" -o simv -l compile.log >/dev/null &&
          ./simv -l sim.log >/dev/null
        ;;
      xsim)
        xvlog --sv "${RTL[@]}" "$tb_file" >compile.log 2>&1 &&
          xelab -timescale 1ns/1ps "$tb" -s "${tb}_snap" >>compile.log 2>&1 &&
          xsim "${tb}_snap" -R >sim.log 2>&1
        ;;
      iverilog)
        iverilog -g2012 -s "$tb" -o sim.vvp "${RTL[@]}" "$tb_file" >compile.log 2>&1 &&
          vvp -n sim.vvp >sim.log 2>&1
        ;;
      *)
        echo "unknown simulator: $SIM" >&2
        exit 2
        ;;
    esac
  )
  rc=$?

  if [ "$rc" -eq 0 ] && [ -f "$log" ] &&
     ! grep -qiE '(^|[^a-z_])(fatal|error)[^a-z_]|\[FAIL\]' "$log"; then
    echo "[PASS] $tb"
    pass=$((pass + 1))
  else
    echo "[FAIL] $tb  (see ${work#"$ROOT"/})"
    fail=$((fail + 1))
    failed+=("$tb")
  fi
done

echo
echo "passed: $pass  failed: $fail"
[ "$fail" -eq 0 ]

#!/usr/bin/env bash
set -euo pipefail

launcher="$1"
par_file="$2"
num_processes="$3"
cimode="$4"

log="eeTurbPipe.reduction.log"
rm -f "$log"

set +e
"$launcher" "$par_file" "$num_processes" --cimode "$cimode" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e

if [[ $status -ne 0 ]]; then
  echo "eeTurbPipe reduction check FAILED: solver exited with status $status"
  exit "$status"
fi

last_diag=$(grep 'eeTurbPipe: step=.*ug-ul relL2=' "$log" | tail -n 1 || true)
if [[ -z "$last_diag" ]]; then
  echo "eeTurbPipe reduction check FAILED: no ug-ul diagnostic was found"
  exit 2
fi

rel_l2=$(sed -n 's/.*ug-ul relL2=\([^ ]*\).*/\1/p' <<<"$last_diag")
max_abs=$(sed -n 's/.*maxAbs=\([^ ]*\).*/\1/p' <<<"$last_diag")

if [[ -z "$rel_l2" || -z "$max_abs" ]]; then
  echo "eeTurbPipe reduction check FAILED: could not parse final diagnostic"
  echo "$last_diag"
  exit 3
fi

# The identical-phase reduction should be much tighter than the flow-solver
# tolerances.  Keep a modest CI margin while still making O(1e-3) drift an
# immediate hard failure.
awk -v r="$rel_l2" -v m="$max_abs" 'BEGIN {
  if ((r + 0.0) >= 1.0e-4 || (m + 0.0) >= 1.0e-4) exit 1;
}' || {
  echo "eeTurbPipe reduction check FAILED: final ug-ul error exceeds 1e-4"
  echo "$last_diag"
  exit 4
}

echo "eeTurbPipe reduction check PASSED: relL2=$rel_l2 maxAbs=$max_abs"

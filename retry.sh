#!/bin/bash
# usage: retry.sh <logfile> <command...> ; re-runs on failure (GPU is shared) until it exits 0
log=$1; shift
for i in $(seq 1 25); do "$@" >> "$log" 2>&1 && exit 0; sleep 90; done

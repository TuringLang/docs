#!/bin/bash

set -euo pipefail

echo "--- Instantiate"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build()'

# Run tutorial
echo "+++ Run tutorial for ${1}"
julia --project=. weave_tutorials.jl "${1}"

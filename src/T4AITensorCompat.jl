#===
T4AITensorCompat.jl

Copyright (c) 2025 Hiroshi Shinaoka and contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
====#

module T4AITensorCompat

import ITensors: ITensors, ITensor, Index, dim, uniqueinds, commoninds, uniqueind
import ITensorMPS
import ITensors: Algorithm, @Algorithm_str
import LinearAlgebra

# Export public API
export TensorTrain
export MPS, MPO, AbstractMPS  # Temporary aliases for migration
export contract
#export fit  # Fit function for summing multiple tensor trains with coefficients
export truncate, truncate!
export maxlinkdim, siteinds
export linkinds, linkind, findsite, findsites, isortho, orthocenter  # Functions for compatibility
export default_maxdim, default_cutoff, default_nsweeps
export lognorm  # Log norm function
export random_mps, random_mpo  # Random tensor train generation
export product  # Official API name (match ITensorMPS)
export apply    # Backwards-compatible alias


include("defaults.jl")
include("tensortrain.jl")

# Temporary type aliases for migration (these map to TensorTrain)
# Defined after TensorTrain is loaded
const MPS = TensorTrain
const MPO = TensorTrain
const AbstractMPS = TensorTrain

include("contraction.jl")
include("random.jl")  # Random tensor train generation functions
include("itensormps_compat.jl")  # Compatibility functions for ITensorMPS API

end

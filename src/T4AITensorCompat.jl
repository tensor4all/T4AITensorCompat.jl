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

import ITensors: ITensors, ITensor, Index, dim, uniqueinds
import ITensorMPS
import ITensors: Algorithm, @Algorithm_str
import LinearAlgebra

# Export public API
export TensorTrain
export contract
export truncate, truncate!
export maxlinkdim, siteinds
export default_maxdim, default_cutoff, default_nsweeps


include("defaults.jl")
include("tensortrain.jl")
include("contraction.jl")

end

#===
defaults.jl - Default parameter values

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

"""
    default_maxdim()

Return the default maximum bond dimension for tensor train operations.

The default is `typemax(Int)`, which effectively means no limit on bond dimension.

# Returns
- `Int`: The default maximum bond dimension
"""
default_maxdim() = typemax(Int)

"""
    default_cutoff()

Return the default cutoff threshold for truncating small singular values.

The default is `1e-30`, which is a very small threshold suitable for high-precision calculations.

# Returns
- `Float64`: The default cutoff value
"""
default_cutoff() = 1e-30

"""
    default_nsweeps()

Return the default number of sweeps for variational fitting algorithms.

The default is `1`, which performs a single sweep.

# Returns
- `Int`: The default number of sweeps
"""
default_nsweeps() = 1
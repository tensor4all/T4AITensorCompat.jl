```@meta
CurrentModule = T4AITensorCompat
```

# T4AITensorCompat

This is the documentation for [T4AITensorCompat](https://github.com/tensor4all/T4AITensorCompat.jl).

A simple tensor train (matrix product state/operator) data structure and algorithms built on top of ITensors.jl.

# API Reference

## Tensor Train Type

```@docs
TensorTrain
TensorTrain(::Vector{ITensor})
TensorTrain(::ITensorMPS.MPS)
TensorTrain(::ITensorMPS.MPS, ::Int, ::Int)
TensorTrain(::ITensorMPS.MPO)
TensorTrain(::ITensorMPS.MPO, ::Int, ::Int)
```

## Tensor Train Operations

```@docs
contract
truncate
truncate!
maxlinkdim
siteinds
linkinds
linkind
findsite
findsites
isortho
orthocenter
fit
lognorm
```

## Default Parameters

```@docs
default_maxdim
default_cutoff
default_nsweeps
```
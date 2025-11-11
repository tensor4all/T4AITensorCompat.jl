# Extension module to resolve ambiguities with ChainRulesCore and InitialValues
# This module is automatically loaded when ChainRulesCore or InitialValues are available

module T4AITensorCompatChainRulesCoreExt

using ChainRulesCore: NoTangent, AbstractThunk, NotImplemented, ZeroTangent
using InitialValues: NonspecificInitialValue, SpecificInitialValue
using ..T4AITensorCompat: TensorTrain

# Resolve ambiguities with ChainRulesCore types
Base.:+(stt::TensorTrain, ::NoTangent) = error("Cannot add ChainRulesCore.NoTangent to TensorTrain")
Base.:+(stt::TensorTrain, ::AbstractThunk) = error("Cannot add ChainRulesCore.AbstractThunk to TensorTrain")
Base.:+(stt::TensorTrain, ::NotImplemented) = error("Cannot add ChainRulesCore.NotImplemented to TensorTrain")
Base.:+(stt::TensorTrain, ::ZeroTangent) = error("Cannot add ChainRulesCore.ZeroTangent to TensorTrain")

# Resolve ambiguities with InitialValues types
Base.:+(stt::TensorTrain, ::Union{NonspecificInitialValue, SpecificInitialValue{typeof(+)}}) = 
    error("Cannot add InitialValues to TensorTrain")

end


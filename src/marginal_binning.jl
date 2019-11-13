# helper functions and types to deal with the fact that we are discretizing the output variables.

struct BinnedCalibrator{H<:Histogram{<:Any, 1,<:Any}}
    hist::H
    Q::Vector{Float64}
    Qo::Float64 #offset
end

BinnedCalibrator(hist, Q) = BinnedCalibrator(hist, Q, zero(Float64))

function (calib::BinnedCalibrator)(x)
    idxs = StatsBase.binindex(calib.hist, x)
    calib.Q[idxs] + calib.Qo
end

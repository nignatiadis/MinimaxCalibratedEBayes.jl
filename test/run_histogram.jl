using MinimaxCalibratedEBayes
using Test

xs = [-Inf; collect(-3.0:0.5:3.0); +Inf]
ws = Float64.(collect(1:14))
tmp_hist = Histogram(xs, ws, :right, false)

tmp_calib = BinnedCalibrator(tmp_hist, ws, 1.0)





@test tmp_calib(-4) == 2.00
@test tmp_calib.([-4, 4]) == [2.0, 15.0]
@test tmp_calib.([-4, -3, -2.9, 4]) == [2.0, 2.0, 3.0, 15.0]


StatsBase.binindex(tmp_hist, -2.999)
, [-2.999, -3.1])

isa(tmp_hist, Histogram)




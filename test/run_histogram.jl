using MinimaxCalibratedEBayes
using Test

xs = range(-3,3;step=0.5)
ws = Float64.(collect(1:14))
tmp_hist = MCEBHistogram(xs)

tmp_calib = BinnedCalibrator(tmp_hist, ws, 1.0)

tmp_calib(-4)
@test tmp_calib(-4) == 2.00
@test tmp_calib.([-4, 4]) == [2.0, 15.0]
@test tmp_calib.([-4, -3, -2.9, 4]) == [2.0, 2.0, 3.0, 15.0]

@test step(tmp_hist) == step(xs)
@test first(tmp_hist) == first(xs)
@test last(tmp_hist) == last(xs)

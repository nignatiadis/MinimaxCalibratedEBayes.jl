# Main entry point to the function

# MCEB +-

# mceb(DiscretizedNormalSamples, Target;
#        PriorFamily = ,
#        alpha=0.1,
#        nbhood=nothing)



# -


Base.@kwdef mutable struct MinimaxCalibratorSetup{T<:EBayesTarget,
                                           DS <: DiscretizedStandardNormalSamples,
                                           Gcal <: ConvexPriorClass,
                                           DT <: DeltaTuner}
    Zs_test
    Zs_train
    target::T
    prior_class::Gcal
    Zs_discr_test::DS
    delta_tuner::DT
    pilot
end






#
# function CEB_ci(Xs_train, Xs_test,
#             ds::MixingNormalConvolutionProblem,
#             target::PosteriorTarget,
#             M_max_num::EmpiricalBayesEstimator, #
#             M_max_denom::EmpiricalBayesEstimator=M_max_num;
#             f_nb::BinnedMarginalDensityNeighborhood = fit(BinnedMarginalDensityNeighborhood, Xs_train, ds),
#             C=:auto, conf=0.9, kwargs...)
#
#
#     m_train = length(Xs_train)
#     m_test = length(Xs_test)
#
#     marginal_grid = ds.marginal_grid
#     marginal_h = ds.marginal_h
#
#
#     num_res = estimate(M_max_num, target.num, Xs_train)
#     denom_res = estimate(M_max_denom, target.denom, Xs_train)
#
#     #TODO: Check if denominator not sure to be >0... abort or throw warning
#     est_target = num_res/denom_res
#
#     calib_target = CalibratedNumerator(target.num, est_target)
#             # Test: Use the Donoho calibrator on the learned function
#
#     #TODO : Change to KWarg..
#
#     if C==:auto
#         C = f_nb.C_std*(1+f_nb.η_infl) + f_nb.C_bias
#     end
#
#     ma = MinimaxCalibrator(ds, f_nb.f, m_test, calib_target; C=C, kwargs...)
#
#     QXs =  ma.(Xs_test)
#     sd = std(QXs)/sqrt(m_test)
#     max_bias = ma.max_bias
#     zz =  get_plus_minus(max_bias, sd, conf)
#
#     zz = zz/denom_res
#     calib_target = est_target + mean(QXs)/denom_res
#     ci_left = calib_target - zz
#     ci_right = calib_target + zz
#
#     don_ci = CalibratedCI(num_res,
#                 M_max_num,
#                 denom_res,
#                 M_max_denom,
#                 est_target,
#                 calib_target,
#                 ci_left,
#                 ci_right,
#                 sd/denom_res[1],
#                 max_bias/denom_res[1],
#                 f_nb,
#                 ma)
#
#     return don_ci
# end
#
# function CEB_ci(Xs_train, Xs_test,
#             ds::MixingNormalConvolutionProblem,
#             target::PosteriorTarget;
#             C=:auto,
#             conf=0.9,
#             kwargs...)
#     # fix this
#     m_train = length(Xs_train)
#     m_test = length(Xs_test)
#
#     marginal_grid = ds.marginal_grid
#     marginal_h = ds.marginal_h
#
#     M_bd = marginal_h/sqrt(2*pi)
#
#     f_const = BinnedMarginalDensity(M_bd, marginal_grid, marginal_h)
#     # Train
#     # Estimate the numerator
#     M_max_num = MinimaxCalibrator(ds, f_const, m_train, target.num;
#                  C=Inf, kwargs...);
#
#
#     # Estimate the denominator
#     M_max_denom = MinimaxCalibrator(ds, f_const, m_train, target.denom;
#                 C=Inf, kwargs...);
#
#
#     CEB_ci(Xs_train, Xs_test, ds, target,
#                 M_max_num,
#                 M_max_denom;
#                 C=C, conf=conf, kwargs...)
#
# end

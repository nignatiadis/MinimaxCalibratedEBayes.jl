function kullback_leibler_project(prior_class::ConvexPriorClass,
	                              Zs_discr::DiscretizedStandardNormalSamples,
								  prob_vec)

								  mproj = Model(prior_class.solver)

								  π = add_prior_variables!(mproj, prior_class)
								  f = MCEB.marginalize(cvx_class, Zs_discr, π)

								  @variable(mproj, entropy_upper_bd_vec[1:length(f)])

								  for i=1:length(f)
								          @constraint(mproj,
								          [-entropy_upper_bd_vec[i]; f[i]; prob_vec[i]] in MathOptInterface.ExponentialCone())
								  end

								  @objective(mproj, Min, sum(entropy_upper_bd_vec ))

								  optimize!(mproj)

								  π_val = JuMP.value.(π)
								  prior_class(π_val)
end

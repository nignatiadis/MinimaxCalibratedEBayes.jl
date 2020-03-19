function npmle(prior_class::ConvexPriorClass,
	                              Zs_discr::DiscretizedStandardNormalSamples)

								  mproj = Model(prior_class.solver)

								  π = add_prior_variables!(mproj, prior_class)
								  f = marginalize(prior_class, Zs_discr, π)

								  wts = Zs_discr.mhist.hist.weights
								  @variable(mproj, ll_lower_bound[1:length(f)])

								  for i=1:length(f)
								          @constraint(mproj,
								          [ll_lower_bound[i]; 1.0; f[i]] in MathOptInterface.ExponentialCone())
								  end

								  @objective(mproj, Max, dot(wts, ll_lower_bound))

								  optimize!(mproj)

								  π_val = JuMP.value.(π)
								  π_val
end

# make sure not to underplot below 0
# we 
function density_bands_to_ribbons(ys, bands_width)
    lower_lims = ys .- max.(ys .- bands_width, 0)
    upper_lims = copy(lower_lims)
    upper_lims .= bands_width
    (lower_lims, upper_lims)
end

function limits_to_ribbons(ys, bands::AbstractVector{<:Tuple})
    bands_left =  ys .- first.(bands)
    bands_right = last.(bands) .- ys
    (bands_left, bands_right)
end 




@userplot ConfIntPlot


@recipe function f(c::ConfIntPlot)

      if length(c.args) == 2
            targets, cis = c.args
            estimated_targets = mean.(cis)
      elseif length(c.args) == 3
             targets, cis, estimated_targets = c.args
      else
             error("Expected 2 or 3 arguments")
      end

      xs = response.(targets)

      left_ci = first.(cis)
      right_ci = last.(cis)

      lower_lims = estimated_targets .- left_ci
      upper_lims = right_ci .- estimated_targets

      ribbons --> (lower_lims, upper_lims)
      fillalpha --> 0.4


      xlab --> L"\mu"
      ylab --> pretty_label(targets[1])
      
      xs, estimated_targets
end


# to plot results from real data analyses.

@recipe function f(resvec::Vector{<:CalibratedMinimaxEstimator})
	targets = [res.target for res in resvec]
	target_grid = response.(targets)
	
	calib_estimates = estimate.(targets, resvec)
	cis = confint.(targets, resvec)
	cis_ribbon  = limits_to_ribbons(calib_estimates, cis)
	pilots = [res.pilot for res in resvec]
	
	@series begin
		seriestype  -->  :path
		color --> "#550133"
		linestyle --> :dot
		label --> L"Pilot estimator $\bar{\theta}(x)$"
		target_grid, pilots
	end 
	
	color --> "#018AC4"
	fillalpha --> 0.39
	ribbon --> cis_ribbon	
	xlabel --> L"x"
	ylabel --> pretty_label(targets[1])
	ylims --> extrema(targets[1])
	legend --> :topleft
	label --> L"Calibrated estimator $\bar{\theta}(x) + \hat{\Delta}(x)$"

	target_grid, calib_estimates
end 
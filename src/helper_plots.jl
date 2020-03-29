function density_bands_to_ribbons(ys, bands)
    lower_lims = ys .- max.(ys .- bands, 0)
    upper_lims = copy(lower_lims)
    upper_lims .= c
    (lower_lims, upper_lims)
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

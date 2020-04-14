using Literate
using Weave


# https://github.com/baggepinnen/LiterateWeave.jl/
function literateweave(source, args...; kwargs...)
    tmpname = tempname()
    Literate.markdown(source, tmpname, documenter=false)
    if source[end-1:end] == "jl"
      sourcename = source[1:end-2] * "md"
    else
      @error "Need .jl file!"
    end
    sourcename = match(r"(\w+.md)", sourcename)[1]
    sourcename = joinpath(tmpname,sourcename)
    jmdsource = replace(sourcename,".md"=>".jmd")
    run(`cp $(sourcename) $(jmdsource)`)
    weave(jmdsource, args...; kwargs...)
end

tmp_dir = @__DIR__
#Literate.markdown(joinpath(tmp_dir,"linear_estimation.jl"), tmp_dir,
#                   documenter=false)


literateweave(joinpath(tmp_dir,"linear_estimation.jl"), out_path=tmp_dir)
literateweave(joinpath(tmp_dir,"data_analysis.jl"), out_path=tmp_dir)

# HACK: So Weave.jl has a submodule `WeavePlots` which is loaded using Requires.jl if Plots.jl is available.
# This means that if we want to overload methods in that submodule we need to wait until `Plots.jl` has been loaded.

# HACK
function Weave.WeavePlots.add_plots_figure(report::Weave.Report, plot::Plots.AnimatedGif, ext)
    chunk = report.cur_chunk
    full_name, rel_name = Weave.get_figname(report, chunk, ext = ext)

    # A `AnimatedGif` has been saved somewhere temporarily, so make a copy to `full_name`.
    cp(plot.filename, full_name; force = true)
    push!(report.figures, rel_name)
    report.fignum += 1
        return full_name
end

function Base.display(report::Weave.Report, m::MIME"text/plain", plot::Plots.AnimatedGif)
    Weave.WeavePlots.add_plots_figure(report, plot, ".gif")
end

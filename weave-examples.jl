# Get paths.
markdown_path = joinpath(@__DIR__, "markdown")

function polish_latex(path::String)
    txt = open(f -> read(f, String), path)
    open(path, "w+") do f
        txt = replace(txt, raw"$$" => raw"\$\$")

        beg_align = r"\\\$\\\$\n?\\begin{align}"
        end_align = r"\\end{align}\n?\\\$\\\$?"

        txt = replace(txt, beg_align => "\$\$\n\\begin{align}")
        txt = replace(txt, end_align => "\\end{align}\n\$\$")

        # Make inline math work.
        inline = r"(?<=[^\\])\$(?!\$)"
        txt = replace(txt, inline => raw"$$")

        # If there's more than three dollar signs together, use only two.
        extra_dollars = r"\${3,}"
        txt = replace(txt, extra_dollars => raw"$$")

        write(f, txt)
    end
end

function add_yaml(path)
    lines = readlines(path, keep=true)

    title = ""
    permalink = "/:collection/:name/"
    fig_dir = replace(basename(path), ".md" => "_files")
    has_yaml = lines[1] == "---\n"


    # Find the ID sections.
    if !has_yaml
        for i = 1:length(lines)
            line = lines[i]

            if startswith(line, "# ")
                title = replace(line, "# " => "")
                title = replace(title, "\n" => "")

                lines[i] = ""
                break
            end
        end
    end

    yaml_lines = ["---\n", "title: $title\n", "permalink: $permalink\n", "---\n"]

    # Write lines back.
    open(path, "w+") do f
        # Prepend YAML.
        if !has_yaml
            for line in yaml_lines
                write(f, line)
            end
        end

        # Figure regex
        fig_regex = Regex("\\!\\[(.*?)\\]\\($fig_dir")
        substitution_string = SubstitutionString("![\\1](/tutorials/$fig_dir")

        # Add remaining content.
        for line in lines
            line = replace(line, r"!\[*\]\(figure" => "![](/tutorials/figure")
            line = replace(line, fig_regex => substitution_string)

            write(f, line)
        end
    end
end

function handle_file(fn)
    polish_latex(fn)
    add_yaml(fn)
end

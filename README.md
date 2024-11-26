# Documentation and Tutorials for Turing.jl

This repository is part of [Turing.jl's](https://turinglang.org/) website (i.e. `https://turinglang.org/docs/`). It contains the Turing.jl documentation and tutorials. 
- The `master` branch contains the quarto source 
- The `gh-pages` branch contains the `html` version of these documents compiled from the `master` branch.

> [!NOTE]  
> Due to [an incompatibility between Bijectors.jl and
> Enzyme.jl](https://github.com/TuringLang/Bijectors.jl/pull/341), the docs
> currently must be built with Julia 1.10. You can do this either by making
> Julia 1.10 your default Julia version (`juliaup default 1.10`), or by
> explicitly specifying Julia 1.10 when running Quarto:
>
> ```bash
> QUARTO_JULIA=$(julia +1.10 -e "println(Sys.BINDIR)")/julia quarto render
> ```

## Local development

To get started with the docs website locally, you'll need to have [Quarto](https://quarto.org/docs/download/) installed.
Make sure you have at least version 1.5 of Quarto installed, as this is required to correctly run [the native Julia engine](https://quarto.org/docs/computations/julia.html#using-the-julia-engine).
Ideally, you should use Quarto 1.6.31 or later as this version fixes [a bug which causes random number generation between different cells to not be deterministic](https://github.com/TuringLang/docs/issues/533).
Note that as of October 2024, Quarto 1.6 is a pre-release version, so you may need to install it from source rather than via a package manager like Homebrew.

Once you have Quarto installed, you can follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/TuringLang/docs
    ```

2. Navigate into the cloned directory:

    ```bash
    cd docs
    ```

3. Instantiate the project environment:

    ```bash
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    ```

4. Preview the website using Quarto Preview:

    ```bash
    quarto preview
    ```

    This will launch a local server at http://localhost:4200/, which you can view in your web browser by navigating to the link shown in your terminal.
    Note: Avoid clicking links in the navbar while previewing locally because they will eventually lead to https links online!

5. Render the website locally:

    ```bash
    quarto render
    ```

    This will build the entire documentation and place the output in the `_site` folder.
    You can then view the rendered website by launching a HTTP server from that directory, e.g. using Python:

    ```bash
    cd _site
    python -m http.server 8000
    ```

    Then, navigate to http://localhost:8000/ in your web browser.

    Note that rendering the entire documentation site can take a long time (usually multiple hours).
    If you wish to speed up local rendering, there are two options available:

    - Download the most recent `_freeze` folder from the [GitHub releases of this repo](https://github.com/turinglang/docs/releases), and place it in the root of the project.
      This will allow Quarto to reuse the outputs of previous computations for any files which have not been changed since that `_freeze` folder was created.

    - Alternatively, render a single tutorial or `qmd` file without compiling the entire site.
      To do this, pass the `qmd` file as an argument to `quarto render`:

      ```
      quarto render path/to/index.qmd
      ```

## Troubleshooting build issues

As described in the [Quarto docs](https://quarto.org/docs/computations/julia.html#using-the-julia-engine), Quarto's Julia engine uses a worker process behind the scenes.
Sometimes this can result in issues with old package code not being unloaded (e.g. when package versions are upgraded).
If you find that Quarto's execution is failing with errors that aren't reproducible via a normal REPL, try adding the `--execute-daemon-restart` flag to the `quarto render` command:

```bash
quarto render /path/to/index.qmd --execute-daemon-restart
```

And also, kill any stray Quarto processes that are still running (sometimes it keeps running in the background):

```bash
pkill -9 -f quarto
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

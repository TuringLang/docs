# Documentation and Tutorials for Turing.jl

This repository is part of [Turing.jl's](https://turinglang.org/) website (i.e. `https://turinglang.org/docs/`). It contains the Turing.jl documentation and tutorials. 
- The `master` branch contains the quarto source 
- The `gh-pages` branch contains the `html` version of these documents compiled from the `master` branch.

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

    This will render the full website in `_site` folder.

    It is also possible to render a single tutorial or `qmd` file without compiling the entire site. This is often helpful to speed up compilation when editing a single docs page. To do this, pass the `qmd` file as an argument to `quarto render`:

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

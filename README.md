# Documentation and Tutorials for Turing.jl

This repository hosts the code for the main Turing.jl documentation `https://turinglang.org/docs/`.
It contains the Turing.jl documentation and tutorials. 

- The `main` branch contains the Quarto source.
- The `gh-pages` branch contains the `html` version of these documents compiled from the `main` branch.

## Local development

To get started with the docs website locally, you'll need to have [Quarto](https://quarto.org/docs/download/) installed.
Make sure you have at least version 1.6.31 of Quarto installed, as this version contains a fix for [a bug where random number generation in different cells was not deterministic](https://github.com/TuringLang/docs/issues/533).

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

4. Preview the website using Quarto.

   > [!WARNING]  
   > This will take a _very_ long time, as it will build every tutorial from scratch. See below for ways to speed this up.

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

## Faster rendering

Note that rendering the entire documentation site can take a long time (usually multiple hours).
If you wish to speed up local rendering, there are two options available:

1. Render a single tutorial or `qmd` file without compiling the entire site.
   To do this, pass the `qmd` file as an argument to `quarto render`:

   ```
   quarto render path/to/index.qmd
   ```

   (Note that `quarto preview` does not support this single-file rendering.)

2. Download the most recent `_freeze` folder from the [GitHub releases of this repo](https://github.com/turinglang/docs/releases), and place it in the root of the project.
   The `_freeze` folder stores the cached outputs from a previous build of the documentation.
   If it is present, Quarto will reuse the outputs of previous computations for any files for which the source is unchanged.

   Note that the validity of a `_freeze` folder depends on the Julia environment that it was created with, because different package versions may lead to different outputs.
   In the GitHub release, the `Manifest.toml` is also provided, and you should also download this and place it in the root directory of the docs.
   
   If there isn't a suitably up-to-date `_freeze` folder in the releases, you can generate a new one by [triggering a run for the `create_release.yml` workflow](https://github.com/TuringLang/docs/actions/workflows/create_release.yml).
   (You will need to have the appropriate permissions; please create an issue if you need help with this.)

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

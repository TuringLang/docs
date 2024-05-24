# Documentation and Tutorials for Turing.jl

This repository's `master` branch contains the quarto source for the Turing.jl documentation and tutorials. The `gh-pages` branch contains the compiled `html` version of these documents. 

To get started with the docs website locally, you'll need to have the following prerequisite installed:

- [Quarto Pre-release](https://quarto.org/docs/download/)

Once you have prerequisite installed, you can follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/TuringLang/docs
    ```

2. Navigate into the cloned directory:

    ```bash
    cd docs
    ```

3. Preview the website using Quarto Preview:

    ```bash
    quarto preview
    ```
This will launch a local server at http://localhost:4200/, which you can view in your web browser by navigating to the link shown in your terminal.
Note: Avoid clicking links in navbar while previewing locally because they will eventually lead to https links online!

4. Render the website locally:

    ```bash
    quarto render
    ```
This will render the full website in `_site` folder.

his repository is part of [Turing.jl's](https://turinglang.org/) Website, i.e. `https://turinglang.org/docs/`. You can see the main website's repo [here](https://github.com/TuringLang/turinglang.github.io).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

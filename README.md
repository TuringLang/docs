# Turing.jl: Docs and Tutorials

This repository contains the quarto documents for the Turing.jl docs and tutorials.

To get started with the docs website locally, you'll need to have the following prerequisites installed:

- [Quarto Pre-release](https://quarto.org/docs/download/)

Once you have these prerequisites installed, you can follow these steps:

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

This repository is another part of [Turing.jl's](https://turinglang.org/) Website, you can see the main website's repo here.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Documentation and Tutorials for Turing.jl

This repository is part of [Turing.jl's](https://turinglang.org/) website (i.e. `https://turinglang.org/docs/`). It contains the Turing.jl documentation and tutorials. 

- The `master` branch contains the quarto source 
- The `gh-pages` branch contains the `html` version of these documents compiled from the `master` branch.

To get started with the docs website locally, you'll need to have the following prerequisite installed:

- [Quarto Pre-release](https://quarto.org/docs/download/)

Once you have the prerequisite installed, you can follow these steps:

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
Note: Avoid clicking links in the navbar while previewing locally because they will eventually lead to https links online!

4. Render the website locally:

    ```bash
    quarto render
    ```
This will render the full website in `_site` folder.

It is also possible to render a single tutorial or `qmd` file without compiling the entire site. This is often helpful to speed up compilation when editing a single docs page. To do this, first `cd` to the file's folder, and run `quarto preview` or `quarto render`: 

```
cd somedocs # This is the folder containing a docs `qmd` file. 
quarto render index.qmd
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

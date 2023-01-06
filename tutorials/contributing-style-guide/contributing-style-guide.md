---
title: Style Guide
permalink: /docs/contributing/style-guide
weave_options:
  error : false
---

# Style Guide

Most Turing code follow the [Invenia](https://invenia.ca/labs/)'s style guide. We would like to thank them for allowing us to access and use it. Please don't let not having read it stop you from contributing to Turing! No one will be annoyed if you open a PR whose style doesn't follow these conventions; we will just help you correct it before it gets merged.


These conventions were originally written at Invenia, taking inspiration from a variety of sources including Python's [PEP8](http://legacy.python.org/dev/peps/pep-0008), Julia's [Notes for Contributors](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md), and Julia's [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/).


What follows is a mixture of a verbatim copy of Invenia's original guide and some of our own modifications.


## A Word on Consistency


When adhering to this style it's important to realize that these are guidelines and not rules. This is [stated best in the PEP8](http://legacy.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds):


> A style guide is about consistency. Consistency with this style guide is important. Consistency within a project is more important. Consistency within one module or function is most important.



> But most importantly: know when to be inconsistent – sometimes the style guide just doesn't apply. When in doubt, use your best judgment. Look at other examples and decide what looks best. And don't hesitate to ask!



## Synopsis


Attempt to follow both the [Julia Contribution Guidelines](https://github.com/JuliaLang/julia/blob/master/CONTRIBUTING.md#general-formatting-guidelines-for-julia-code-contributions), the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/), and this guide. When convention guidelines conflict this guide takes precedence (known conflicts will be noted in this guide).


  * Use 4 spaces per indentation level, no tabs.
  * Try to adhere to a 92 character line length limit.
  * Use upper camel case convention for [modules](https://docs.julialang.org/en/v1/manual/modules/) and [types](https://docs.julialang.org/en/v1/manual/types/).
  * Use lower case with underscores for method names (note: Julia code likes to use lower case without underscores).
  * Comments are good, try to explain the intentions of the code.
  * Use whitespace to make the code more readable.
  * No whitespace at the end of a line (trailing whitespace).
  * Avoid padding brackets with spaces. ex. `Int64(value)` preferred over `Int64( value )`.


## Editor Configuration


### Sublime Text Settings


If you are a user of Sublime Text we recommend that you have the following options in your Julia syntax specific settings. To modify these settings first open any Julia file (`*.jl`) in Sublime Text. Then navigate to: `Preferences > Settings - More > Syntax Specific - User`


```json
{
    "translate_tabs_to_spaces": true,
    "tab_size": 4,
    "trim_trailing_white_space_on_save": true,
    "ensure_newline_at_eof_on_save": true,
    "rulers": [92]
}
```


### Vim Settings


If you are a user of Vim we recommend that you add the following options to your `.vimrc` file.


```
set tabstop=4                             " Sets tabstops to a width of four columns.
set softtabstop=4                         " Determines the behaviour of TAB and BACKSPACE keys with expandtab.
set shiftwidth=4                          " Determines the results of >>, <<, and ==.

au FileType julia setlocal expandtab      " Replaces tabs with spaces.
au FileType julia setlocal colorcolumn=93 " Highlights column 93 to help maintain the 92 character line limit.
```


By default, Vim seems to guess that `.jl` files are written in Lisp. To ensure that Vim recognizes Julia files you can manually have it check for the `.jl` extension, but a better solution is to install [Julia-Vim](https://github.com/JuliaLang/julia-vim), which also includes proper syntax highlighting and a few cool other features.


## Test Formatting


### Testsets


Julia provides [test sets](https://docs.julialang.org/en/v1/stdlib/Test/#Working-with-Test-Sets-1) which allows developers to group tests into logical groupings. Test sets can be nested and ideally packages should only have a single "root" test set. It is recommended that the "runtests.jl" file contains the root test set which contains the remainder of the tests:


```julia
@testset "PkgExtreme" begin
    include("arithmetic.jl")
    include("utils.jl")
end
```


The file structure of the `test` folder should mirror that of the `src` folder. Every file in `src` should have a complementary file in the `test` folder, containing tests relevant to that file's contents.


### Comparisons


Most tests are written in the form `@test x == y`. Since the `==` function doesn't take types into account tests like the following are valid: `@test 1.0 == 1`. Avoid adding visual noise into test comparisons:


```julia
# Yes:
@test value == 0

# No:
@test value == 0.0
```


In cases where you are checking the numerical validity of a model's parameter estimates, please use the `check_numerical` function found in `test/test_utils/numerical_tests.jl`. This function will evaluate a model's parameter estimates using tolerance levels `atol` and `rtol`. Testing will only be performed if you are running the test suite locally or if Travis is executing the "Numerical" testing stage.


Here is an example of usage:


```julia
# Check that m and s are plus or minus one from 1.5 and 2.2, respectively.
check_numerical(chain, [:m, :s], [1.5, 2.2], atol = 1.0)

# Checks the estimates for a default gdemo model using values 1.5 and 2.0.
check_gdemo(chain, atol = 0.1)

# Checks the estimates for a default MoG model.
check_MoGtest_default(chain, atol = 0.1)
```


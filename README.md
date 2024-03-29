# Evaluating Scikit-Learn Models

![Tests](https://github.com/joel-beck/eval-sklearn/actions/workflows/tests.yaml/badge.svg)
![Pre-Commit](https://github.com/joel-beck/eval-sklearn/actions/workflows/pre-commit.yaml/badge.svg)

This repository follows along my learning path of the popular Python library `Scikit-Learn`.
Throughout this project I aim to conform to best practices of the `Scikit-Learn` community in the year 2022 and provide convenience functions for evaluating classification, regression and clustering models.

Further, this project serves as an opportunity to gain familiarity with developing custom Python packages.
As such the `eval_sklearn` package is under active development and (for now) not stable!
External contributions are, however, very welcome.

## Installation

In order to use the `eval_sklearn` package you can clone the repository from GitHub and install all required dependencies with

```
pip install .
```

If you want to contribute to development please run the command

```
pip install -e .[dev]
```

to install the `eval_sklearn` package in *editable mode* with all development dependencies.

Finally, in order to run the practical examples in the `notebooks` folder, you can install the full set of dependencies with

```
pip install -e .[dev, notebooks]
```

**Note**:
Running the notebooks additionally requires the `lightgbm` package whose installation via `pip` is unreliable at the time of writing.
Thus, I recommend installing this package via `conda` with the command

```
conda install -c conda-forge lightgbm=3.3
```

If you do not want to use conda and installation with `pip` fails, the [official installation guide](https://github.com/microsoft/LightGBM/tree/master/python-package) provides help for all operating systems.


## Resources

Apart from Stackoverflow rabbit holes and various blog posts this project is mainly build upon the two excellent textbooks (with accompanying video lectures)

- Sebastian Raschka et al.: Machine Learning with PyTorch and Scikit-Learn (2022)
    - [Youtube Lectures from 2020](https://www.youtube.com/playlist?list=PLTKMiZHVd_2KyGirGEvKlniaWeLOHhUF3)
- Andreas Müller, Sarah Guido: Introduction to Machine Learning with Python (2016)
    - [Youtube Lectures from 2020](https://www.youtube.com/playlist?list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM)

and the fantastic [Scikit Learn Documentation](https://scikit-learn.org/stable/).

Since this repository primarily serves as my own reference, it will certainly lack extensive documentation.
I invite you to scroll through the code nonetheless to discover some `scikit-learn` gems that might be useful in your own projects.
Happy coding!

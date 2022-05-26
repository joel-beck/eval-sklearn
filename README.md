# Evaluating Scikit-Learn Models

This repository follows along my learning path of the popular Python library `Scikit-Learn`.
Throughout this project I aim to conform to best practices of the `Scikit-Learn` community in the year 2022 and provide convenience functions for evaluating classification, regression and clustering models.

Further, this project serves as an opportunity to gain familiarity with developing custom Python packages.
As such this package is under active development and (for now) not stable!
External contributions are, however, very welcome.

All code will be reproducible. For this purpose a `environment.yml` file is provided to install all required packages into a `conda` environment, which can be created with the command
<!--
TODO: Replace the environment.yml file with a proper dependency collection in setup.cfg
-->
```
conda env create -f environment.yml
```

Apart from `scikit-learn` itself, I will make use of the related packages `numpy` and `pandas` for data manipulation, `matplotlib` and `seaborn` for visualization as well as `xgboost` and `lightgbm` for additional modeling based on Gradient Boosting.

Apart from [Stackoverflow](https://stackoverflow.com/questions/tagged/scikit-learn) rabbit holes and various blog posts this project is mainly build upon the two excellent textbooks (with accompanying video lectures)

- Sebastian Raschka et al.: Machine Learning with PyTorch and Scikit-Learn (2022)
    - [Youtube Lectures from 2020](https://www.youtube.com/playlist?list=PLTKMiZHVd_2KyGirGEvKlniaWeLOHhUF3)
- Andreas Müller, Sarah Guido: Introduction to Machine Learning with Python (2016)
    - [Youtube Lectures from 2020](https://www.youtube.com/playlist?list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM)

and the fantastic [Scikit Learn Documentation](https://scikit-learn.org/stable/).

Since this repository primarily serves as my own reference, it will certainly lack extensive documentation and explanations.
I invite you to scroll through the code nonetheless to discover some `scikit-learn` gems that might be useful in your own projects.
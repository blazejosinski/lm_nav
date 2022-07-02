# Source code of LM-Nav

<a href="http://colab.research.google.com/github/blazejosinski/lm_nav/blob/main/colab_experiment.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a> 

To run locally, install the package:

```pip install .```

The code was tested with python 3.7.13. It assumes access to GPU and CUDA 10.2 is installed.

To run experiments on one of the environments use jupyter notebook and open `jupyter_experiment.ipynb`.

We are providing a cached version of the OpenAI API calls for the sample queries. If you also want to run the GPT-3 part of the pipeline, you need to provide OpenAI API key (see the docs out the API docs [here](https://openai.com/api/)), and pass it the OpenAI API, e.g.:

``OPENAI_API_KEY=sk-[]  jupyter notebook``

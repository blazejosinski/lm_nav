# Source code of LM-Nav

To prepare the environment, run:

```pip install .```

The code was tested with python 3.7.13. It assumes access to GPU and CUDA 10.2 is installed.

To run experiments on one of the environments use jupyter notebook:

``OPENAI_API_KEY=sk-[]  jupyter notebook``

The OPENAI_API_KEY is required if you want to run the GPT-3 part of the pipeline. check out the API docs [here](https://openai.com/api/). We are providing a cached version of the API calls for some sample queries in the Jupyter notebook.

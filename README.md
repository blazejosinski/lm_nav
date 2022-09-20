# Source code of LM-Nav

<a href="http://colab.research.google.com/github/blazejosinski/lm_nav/blob/main/colab_experiment.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=22.5></a> 

THIS SOFTWARE AND/OR DATA WAS DEPOSITED IN THE BAIR OPEN RESEARCH COMMONS REPOSITORY ON JULY 11, 2022.

## Introduction

This repository contains code used in *LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action* by Dhruv Shah, Błażej Osiński, Brian Ichter, and Sergey Levine.

In order to dive into the code, we suggest starting from the following:

* `jupyter_experiment.ipynb` - notebook to run the LM-Nav pipeline for text queries on two different graphs. This includes running GPT3, CLIP, and our custom graph search algorithm.
* `colab_experiment.ipynb` - colab version of the above notebook. You can easily [run it in your browser](http://colab.research.google.com/github/blazejosinski/lm_nav/blob/main/colab_experiment.ipynb)!
* `ablation_text_to_landmark.ipynb` - notebook with ablation experiments for the language processing part: comparing GPT3 to open-source alternatives and a simple NLP baseline.

## Installation

The code was tested with python 3.7.13. It assumes access to GPU and CUDA 10.2 is installed.

To run locally, install the package:

```pip install .```

Then simply open `jupyter_experiments.ipynb` or `ablation_text_to_landmark.ipynb` in jupyter notebook.

## LLMs APIs

For the LM-Nav pipeline notebooks, we added a cached version of the OpenAI API calls for the sample queries. If you also want to run the GPT-3 part of the pipeline, you need to provide OpenAI API key (see the docs out the API docs [here](https://openai.com/api/)), and pass it the OpenAI API, e.g.:

``OPENAI_API_KEY=sk-[]  jupyter notebook``

Likewise, in order to re-run the ablation experiments with the open-source models you need to specify [GooseAI API key](https://goose.ai/).

## Citation

If you find this work useful, please consider citing:

```
@misc{shah2022lmnav,
      title={LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action}, 
      author={Dhruv Shah and Blazej Osinski and Brian Ichter and Sergey Levine},
      year={2022},
      eprint={2207.04429},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

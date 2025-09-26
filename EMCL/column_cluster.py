"""
Use LLM to embed the columns, form a graph, then max the modularity.
2 groups or set some threshold? make it hyperparam for further experiments.
"""

import torch
import numpy as np
import pandas as pd


def embed_columns(cleaned_data):
    """
    clean_data: The clean part of the raw data.
    output: a graph, each node is a column, each edge is the similarity between two columns.
    “”“
    pass
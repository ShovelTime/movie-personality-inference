import os
import sys
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (
    map, ndcg_at_k, precision_at_k, recall_at_k
)
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.utils.notebook_utils import store_metadata


print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))


if not os.path.exists("/tmp/dataset/out.xlsx"):
    raise Exception("Dataset is not present! place in the following path: /tmp/dataset/parsed_reviews.xlsx")

train = pd.read_excel("/tmp/dataset/out.xlsx", 'training')
eval = pd.read_excel("/tmp/dataset/out.xlsx", 'eval')

# top k items to recommend
TOP_K = 10

# Model parameters
EPOCHS = 100
BATCH_SIZE = 256

SEED = DEFAULT_SEED  # Set None for non-deterministic results



leave_one_out_test = eval.groupby("user_id").last().reset_index()



train_file = "./train.csv"
test_file = "./test.csv"
leave_one_out_test_file = "./leave_one_out_test.csv"
train.to_csv(train_file, index=False)
eval.to_csv(test_file, index=False)
leave_one_out_test.to_csv(leave_one_out_test_file, index=False)


data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED, overwrite_test_file_full=True)


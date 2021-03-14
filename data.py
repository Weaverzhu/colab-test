import gzip
import os
import pickle
import yaml

from PGPR.data_utils import AmazonDataset
from PGPR.knowledge_graph import KnowledgeGraph
from PGPR.preprocess import generate_labels
from PGPR.utils import save_dataset
from utils import do_or_load as do

config = yaml.load(open("./config/data.yml", "r"), yaml.SafeLoader)


def main():
    # the name of dataset
    dataset = config["data_select"]
    # tmp dir for dataset
    tmp_dir = config["tmp_dir"]
    # dataset_dir
    dataset_dir = config["data_{}".format(dataset)]

    print("[data] dataset = {}".format(dataset))
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    print("[data] creating dataset instance", dataset)
    ds = do(
        lambda: AmazonDataset(dataset_dir),
        tmp_dir + dataset + ".ds",
    )

    print("[data] generate knowledge graph instance")

    def kg_func():
        kg = KnowledgeGraph(ds)
        kg.compute_degrees()
        return kg

    kg = do(kg_func, tmp_dir + dataset + ".kg")

    print("[data] generating labels")
    def label_func():
        def _f(mode):
            review_file = "{}/{}.txt.gz".format(dataset_dir, mode)
            user_products = {}  # {uid: [pid,...], ...}
            with gzip.open(review_file, "r") as f:
                for line in f:
                    line = line.decode("utf-8").strip()
                    arr = line.split("\t")
                    user_idx = int(arr[0])
                    product_idx = int(arr[1])
                    if user_idx not in user_products:
                        user_products[user_idx] = []
                    user_products[user_idx].append(product_idx)
            return user_products

        return [_f('train'), _f('test')]
    labels = do(label_func, label_func)


if __name__ == "__main__":
    main()

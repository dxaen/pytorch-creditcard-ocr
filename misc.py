import torch


def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)

def store_labels(path, labels):
    labels = map(str, labels)
    with open(path, "w") as f:
        f.write("\n".join(labels))

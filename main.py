import os
import pickle
import re
import gzip
import zipfile
import random as pyrandom
import time
from textwrap import indent
import shelve
import urllib3

import PIL.Image
from flax import linen as nn
import jax.random as random
import jax.numpy as jnp
from jax import vmap, grad, jit
from jax.scipy.special import logsumexp

LAYER_SIZES = [28 ** 2, 28 ** 2, 28 ** 2, 3]
STEP_SIZE = 0.01
NUM_EPOCHS = 100
SHOW_GRADIENTS = False  # set to True to see the gradients every epoch. Slow!


def download_data():
    if not os.path.exists("data"):
        os.mkdir("data")
    for fn in ["byhand.zip", "byhand_adaptive.zip", "byhand_interpolative.zip", "byhand_normalized.zip", "whiteonblack.zip"]:
        if not os.path.exists("data/" + fn):
            uri = "https://github.com/clayote/aiml425assignment2/raw/main/data/" + fn
            got = urllib3.request("GET", uri)
            assert got.status == 200, "Network error when downloading " + fn
            with open("data/" + fn, "wb") as f:
                f.write(got.data)
    for fn in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k.labels-idx1-ubyte.gz"]:
        if not os.path.exists("data/" + fn):
            uri = "https://storage.googleapis.com/cvdf-datasets/mnist/" + fn
            got = urllib3.request("GET", uri)
            assert got.status == 200, "Network error when downloading " + fn
            with open("data/" + fn, "wb") as f:
                f.write(got.data)



def preprocess(prefix="train", random_seed=0, holdout_prob=0.):
    rotmap = {-90: 'left', 0: 'none', 90: 'right'}
    randomizer = pyrandom.Random(random_seed)
    if not os.path.exists("preprocessed"):
        os.mkdir("preprocessed")
    if os.path.exists(f"data/{prefix}.zip"):
        with zipfile.ZipFile(f"data/{prefix}.zip") as zf, shelve.open(f"preprocessed/{prefix}_shelf.pkl") as shelf:
            for i, fn in enumerate(zf.namelist()):
                rotation = randomizer.choice([-90, 0, 90])
                match = re.match(r".*(\d)\.pgm", fn)
                assert match, "Couldn't get label for " + fn
                label = match.groups()[0]
                with zf.open(fn) as inf:
                    shelf[f"{fn}_labeled_{label}_rotated_{rotmap[rotation]}"] = jnp.array(
                        PIL.Image.open(inf).rotate(rotation).getdata()
                    )
        return
    with gzip.open(f"data/{prefix}-images-idx3-ubyte.gz") as imgf, gzip.open(
            f"data/{prefix}-labels-idx1-ubyte.gz") as labf:
        magic = imgf.read(4)
        assert magic == b'\x00\x00\x08\x03', f"Test image file magic number was {magic}, should have been 0x00000803"
        magic = labf.read(4)
        assert magic == b'\x00\x00\x08\x01', f"Test label file magic number was {magic}, should have been 0x00000801"
        n_images = int.from_bytes(imgf.read(4), "big")
        assert n_images == int.from_bytes(labf.read(4), "big")
        width = int.from_bytes(imgf.read(4), "big")
        height = int.from_bytes(imgf.read(4), "big")
        print(f"{prefix} dataset has {n_images} images of size {width}x{height}")
        if holdout_prob == 0.:
            holdout = shelve.Shelf({})
        else:
            holdout = shelve.open(f"preprocessed/{prefix}_holdout_shelf.pkl")
        with shelve.open(f"preprocessed/{prefix}_shelf.pkl") as shelf, holdout as holdout_shelf:
            for i in range(n_images):
                do_holdout = randomizer.random() < holdout_prob
                label = int.from_bytes(labf.read(1), "big")
                rotation = randomizer.choice([-90, 0, 90])
                # I could probably do this without PIL...
                img = PIL.Image.frombytes("L", (width, height),
                                         imgf.read(width * height))
                arr = jnp.array(img.rotate(rotation).getdata())
                key = f"image_{i}_labeled_{label}_rotated_{rotmap[rotation]}"
                if do_holdout:
                    holdout_shelf[key] = arr
                else:
                    shelf[key] = arr


# begin jax tutorial
# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key,
                                                                       (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in
            zip(sizes[:-1], sizes[1:], keys)]


params = init_network_params(LAYER_SIZES, random.key(0))


def predict(params, coord):
    activations = coord
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        # Log sigmoid gave spiky inconsistent behavior, similar to before.
        # CeLU had a good-looking spike at the 3rd gen, when the network was 28**3 wide in the middle.
        activations = nn.activation.celu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return grads, [(w - STEP_SIZE * dw, b - STEP_SIZE * db)
                   for (w, b), (dw, db) in zip(params, grads)]


@jit
def apply_grads(params, grads):
    return [(w - STEP_SIZE * dw, b - STEP_SIZE * db)
                   for (w, b), (dw, db) in zip(params, grads)]


accuracies = {"training": [], "testing": []}


def train(num_epochs, train_inputs, train_labels, test_inputs, test_labels,
          start_epoch=0):
    global params, accuracies
    for epoch in range(start_epoch, num_epochs):
        start_time = time.monotonic()
        grads = []
        gradient, params = update(params, train_inputs, train_labels)
        grads.append(gradient)  # in case of batches, later
        epoch_time = time.monotonic() - start_time

        train_acc = accuracy(params, train_inputs, train_labels)
        test_acc = accuracy(params, test_inputs, test_labels)
        accuracies["training"].append(train_acc)
        accuracies["testing"].append(test_acc)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))
        if not os.path.exists("trained"):
            os.mkdir("trained")
        with gzip.open(f"trained/grads{epoch}.pkl.gz", "wb") as outf:
            pickle.dump(grads, outf)
        with gzip.open(f"trained/acc{epoch}.pkl.gz", "wb") as outf:
            pickle.dump(accuracies, outf)
        if SHOW_GRADIENTS:
            print("Gradients:")
            for gradients in grads:
                gradia = []
                for layer in gradients:
                    gradience = []
                    for gradient in layer:
                        if gradient.any():
                            gradience.append(indent(str(gradient), "\t\t"))
                        else:
                            gradience.append("\t\t...")
                    if any(gra for gra in gradience if gra != "\t\t..."):
                        gradia.append("\t(")
                        for gra in gradience:
                            gradia.append(gra)
                        gradia.append("\t)")
                    else:
                        gradia.append("\t(...)")
                if any(g for g in gradia if g != "\t(...)"):
                    print("[")
                    for g in gradia:
                        print(g)
                    print("]")
                else:
                    print("[...]")


# end jax tutorial


onehot = {
    'left': jnp.array([1., 0., 0.]),
    'none': jnp.array([0., 1., 0.]),
    'right': jnp.array([0., 0., 1.])
}


def make_input(prefix="train"):
    input_data = []
    input_labels = []
    shelf = shelve.open(f"preprocessed/{prefix}_shelf.pkl")
    for key, val in shelf.items():
        match = re.match(r".*_labeled_(\d+)_rotated_(\w+)", key)
        assert match, "Bad key: " + key
        label, rotation_s = match.groups()
        input_data.append(jnp.asarray(val))
        input_labels.append(onehot[rotation_s])
    return jnp.array(input_data), jnp.array(input_labels)


def restore_state():
    global params, accuracies
    if not os.path.exists("trained"):
        return -1
    highest_epoch = -1
    for filename in os.listdir("trained"):
        if filename.startswith("grads"):
            epoch = int(filename.removeprefix("grads").removesuffix(".pkl.gz"))
            if epoch > highest_epoch:
                highest_epoch = epoch
    if highest_epoch < 0:
        return -1
    for epoch in range(0, highest_epoch + 1):
        with gzip.open(f"trained/grads{epoch}.pkl.gz", "rb") as inf:
            grads = pickle.load(inf)
        for gradient in grads:
            params = apply_grads(params, gradient)
    with gzip.open(f"trained/acc{highest_epoch}.pkl.gz", "rb") as inf:
        accuracies = pickle.load(inf)
    return highest_epoch


if __name__ == "__main__":
    download_data()
    preprocess("t10k", holdout_prob=0.05)
    for dataset in ("train", "whiteonblack", "byhand", "byhand_adaptive", "byhand_interpolative", "byhand_normalized"):
        preprocess(dataset)
    highest_epoch = restore_state()
    input_data, input_labels = make_input("train")
    test_data, test_labels = make_input("t10k")
    train(NUM_EPOCHS, input_data, input_labels, test_data, test_labels, highest_epoch+1)
    for validation_set in ("t10k_holdout", "whiteonblack", "byhand", "byhand_adaptive", "byhand_interpolative", "byhand_normalized"):
        validation_data, validation_labels = make_input(validation_set)
        validation_acc = accuracy(params, validation_data, validation_labels)
        print("Validation set", validation_set, "has accuracy", validation_acc)

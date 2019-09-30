"""
John F. Wu

Train a deep convnet to predict gas mass fraction using the ALFALFA a.40
data set. Saves the best model in the `{PATH}/models` directory.
"""


from fastai import *
from fastai.callbacks import *
from fastai.vision import *

from glob import glob
import seaborn as sns

PATH = os.path.abspath("..")
sys.path.append(f"{PATH}/src")

from mxresnet import *
from ranger import Ranger


xGASS_stats = [tensor([-0.0169, -0.0105, -0.0004]), tensor([0.9912, 0.9968, 1.0224])]

opt = {
    "seed": 12345,
    "val_pct": 0.2,
    "bs": 32,
    "sz": 224,
    "model": mxresnet34(),
    "n_epochs": 100,
    "lr": 3e-2,
    "tfms": get_transforms(
        do_flip=True,
        flip_vert=True,
        max_zoom=1.0,
        max_rotate=15.0,
        max_lighting=0,
        max_warp=0,
    ),
}


def load_df(all_properties=True):
    """Load the dataframe containing gas mass fractions, and if 
    `all_properties` is set True, then cut on the HI sources that
    also have SFRs and metallicities (from JHU/MPA catalogs).
    """

    if all_properties:
        return pd.read_csv(f"{PATH}/data/a40-SDSS_galaxy-properties.csv")
    else:
        return pd.read_csv(f"{PATH}/data/a40-SDSS_gas-frac.csv")


if __name__ == "__main__":

    # load DataBunch
    df = load_df(all_properties=False)

    src = (
        ImageList.from_df(
            df, path=PATH, folder="images-OC", suffix=".jpg", cols="AGCNr"
        )
        .split_by_rand_pct(opt["val_pct"], seed=opt["seed"])
        .label_from_df(cols=["logfgas"], label_cls=FloatList)
    )

    data = (
        src.transform(tfms, size=opt["sz"])
        .databunch(bs=opt["bs"])
        .normalize(xGASS_stats)
    )

    # reformulate model to output single regression
    model = opt["model"]
    model[-1] = nn.Linear(model[-1].in_features, 1, bias=True).cuda()

    # initialize Fastai learner
    learn = Learner(
        data,
        model=model,
        opt_func=partial(Ranger),
        loss_func=root_mean_squared_error,
        wd=1e-3,
        bn_wd=False,
        true_wd=True,
    )

    # train and keep track of best model
    learn.fit_one_cycle(
        n_epochs=opt["n_epochs"],
        max_lr=opt["lr"],
        callbacks=[SaveModelCallback(learn, every="improvement", name="best_a40")],
    )

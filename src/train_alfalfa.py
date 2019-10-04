"""
John F. Wu

Train a deep convnet to predict gas mass fraction using the ALFALFA a.40
data set. Saves the best model in the `{PATH}/models` directory.
"""


from fastai import *
from fastai.callbacks import *
from fastai.vision import *

from glob import glob
from optparse import OptionParser
import seaborn as sns

PATH = os.path.abspath("..")
sys.path.append(f"{PATH}/src")

from mxresnet import *
from ranger import Ranger


xGASS_stats = [tensor([-0.0169, -0.0105, -0.0004]), tensor([0.9912, 0.9968, 1.0224])]

model = mxresnet34()

tfms = get_transforms(
    do_flip=True,
    flip_vert=True,
    max_zoom=1.0,
    max_rotate=15.0,
    max_lighting=0,
    max_warp=0,
)


def command_line():
    """ Controls the command line argument handling.
    """

    # read in the cmd line arguments
    USAGE = "usage:\t %prog [options]\n"
    parser = OptionParser(usage=USAGE)

    # add options
    parser.add_option("--seed", dest="seed", default=12345, help="random seed (int)")
    parser.add_option("--sz", dest="sz", default=224, help="image size")
    parser.add_option("--val-pct", dest="val_pct", default=0.2, help="validation percentage")
    parser.add_option("--bs", dest="bs", default=32, help="batch size")
    parser.add_option("--fp16", dest="mixed_precision", default=False, help="mixed precision")
    parser.add_option("--n_epochs", dest="n_epochs", default=100, help="number of epochs")
    parser.add_option("--lr", dest="lr", default=3e-2, help="maximum learning rate")
    parser.add_option(
        "--all-properties", 
        dest="all_properties", 
        default=False, 
        help="Load catalog with galaxy SFR and metallicity",
    )

    (options, args) = parser.parse_args()

    return options, args


def load_df(all_properties):
    """Load the dataframe containing gas mass fractions, and if 
    `all_properties` is set True, then cut on the HI sources that
    also have SFRs and metallicities (from JHU/MPA catalogs).
    """

    if all_properties:
        return pd.read_csv(f"{PATH}/data/a40-SDSS_galaxy-properties.csv")
    else:
        return pd.read_csv(f"{PATH}/data/a40-SDSS_gas-frac.csv")


if __name__ == "__main__":
    
    # load options
    opt, args = command_line()

    # load DataBunch
    df = load_df(all_properties=opt.all_properties)

    src = (
        ImageList.from_df(
            df, path=PATH, folder="images-OC", suffix=".jpg", cols="AGCNr"
        )
        .split_by_rand_pct(opt.val_pct, seed=opt.seed)
        .label_from_df(cols=["logfgas"], label_cls=FloatList)
    )

    data = (
        src.transform(tfms, size=opt.sz)
        .databunch(bs=opt.bs)
        .normalize(xGASS_stats)
    )

    # reformulate model to output single regression
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
    
    if opt.mixed_precision:
        learn.to_fp16()
    else:
        learn.to_fp32()

    # train and keep track of best model
    learn.fit_one_cycle(
        cyc_len=opt.n_epochs,
        max_lr=opt.lr,
        callbacks=[SaveModelCallback(learn, every="improvement", name="best_a40")],
    )

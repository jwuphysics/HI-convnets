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

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(f"{PATH}/src")

from mxresnet import *
from ranger import Ranger


xGASS_stats = [tensor([-0.0169, -0.0105, -0.0004]), tensor([0.9912, 0.9968, 1.0224])]

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

    # read in the command line arguments
    parser = OptionParser(usage="usage:\t %prog [options]\n)")

    # add options
    parser.add_option("--seed", dest="seed", type=int, default=12345, help="random seed (int)")
    parser.add_option("--sz", dest="sz", type=int, default=224, help="image size")
    parser.add_option("--val-pct", dest="val_pct", type=float, default=0.2, help="validation percentage")
    parser.add_option("--bs", dest="bs", type=int, default=32, help="batch size")
    parser.add_option("--precision", dest="precision", type=str, default="full", help="full or mixed precision")
    parser.add_option("--n_epochs", dest="n_epochs", type=int, default=100, help="number of epochs")
    parser.add_option("--lr", dest="lr", type=float, default=3e-2, help="maximum learning rate")
    parser.add_option("--model", dest="model", type=str, default="mxresnet50", help="convnet architecture")
    parser.add_option(
        "--catalog",
        dest="catalog",
        default="fgas",
        help="Load catalog with only `fgas`, or `all` galaxy properties"
    )
    parser.add_option(
        "--save",
        dest="save_fname",
        type=str,
        default="best_a40",
        help="destination of best model"
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
    all_properties = (opt.catalog == "all")
    df = load_df(all_properties=all_properties)
    print(f"Loaded `{opt.catalog}` catalog of length {len(df)}")

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

    # select model
    if opt.model in ["mxresnet18", "18"]:
        model = mxresnet18()
    elif opt.model in ["mxresnet34", "34"]:
        model = mxresnet34()
    elif opt.model in ["mxresnet50", "50"]:
        model = mxresnet50()
    elif opt.model in ["mxresnet101", "101"]:
        model = mxresnet101()
    elif opt.model in ["mxresnet152", "152"]:
        model = mxresnet152()
    else:
        sys.exit("Please specify a valid model of the `mxresnet` variant")

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

    if opt.precision == "mixed":
        learn.to_fp16()
    elif opt.precision == "full":
        learn.to_fp32()
    else:
        sys.exit("Please specify mixed or full floating-point precision.")

    # train (do not keep track of best model)
    learn.fit_one_cycle(
            cyc_len=opt.n_epochs,
            max_lr=opt.lr,
        )
    
    if (opt.save_fname != "") and (opt.save_fname.lower() != "none"):
        learn.save(opt.save_fname)
        
        

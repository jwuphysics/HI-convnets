"""
John F. Wu (2019)

Now fetches grz imaging from Legacy Survey for the 
alpha.100 ALFALFA detections.

Default size is now 448x448 (ImageNet-inspired).
"""

from optparse import OptionParser
import pandas as pd
import skimage.io

import os
import time
import sys
import urllib

# assuming that this is being run in the ${ROOT}/src directory
PATH = os.path.abspath("..")


class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data):
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def cmdline():
    """ Controls the command line argument handling for this little program.
    """

    # read in the cmd line arguments
    USAGE = "usage:\t %prog [options]\n"
    parser = OptionParser(usage=USAGE)

    # add options
    parser.add_option(
        "--output",
        dest="output",
        default=f"{PATH}/images-legacy",
        # default=f'{PATH}/images-xgass-legacy',
        help="Path to save image data",
    )
    parser.add_option("--size", dest="size", default=448, help="Default size of images")
    parser.add_option(
        "--pixscale", dest="pixscale", default=0.262, help="Pixel scale of images"
    )
    parser.add_option(
        "--cat",
        dest="cat",
        default=f"{PATH}/data/a100.code12.tab1.180315.csv",
        # default=f'{PATH}/../xgass-convnets/data/xGASS_representative_sample.csv',
        help="Catalog to get image names from.",
    )

    (options, args) = parser.parse_args()

    return options, args


def main():

    opt, arg = cmdline()

    # load the data
    df = pd.read_csv(opt.cat)

    pixscale = opt.pixscale
    size = opt.size

    # remove trailing slash in output path if it's there.
    opt.output = opt.output.rstrip("\/")

    # total number of images
    n_gals = df.shape[0]

    for row in df.itertuples():
        url = (
            "http://legacysurvey.org/viewer/cutout.jpg"
            "?ra={}"
            "&dec={}"
            "&pixscale={}"
            "&layer=dr8"
            "&size={}".format(row.RAdeg_OC, row.DECdeg_OC, pixscale, size)
        )
        if not os.path.isfile(f"{opt.output}/{row.AGCNr}.jpg"):
            try:
                img = skimage.io.imread(url)
                skimage.io.imsave(f"{opt.output}/{row.AGCNr}.jpg", img)
                time.sleep(0.1)
            except urllib.error.HTTPError:
                pass
        current = row.Index / n_gals * 100
        status = "{:.3f}% of {} completed.".format(current, n_gals)
        Printer(status)

    print("")


if __name__ == "__main__":
    main()

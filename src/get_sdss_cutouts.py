"""
John F. Wu (2019)
Initially based on Steven Boada's script to get gri imaging for SDSS
spectroscopic targets.

Now fetches gri imaging for the alpha.100 ALFALFA detections.

Default size is now 224x224 (ImageNet-inspired) and we do not rescale
the pixel sizes from the native SDSS resolution.
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
        default=f"{PATH}/images-nibles",
        help="Path to save image data",
    )
    parser.add_option(
        "--width", dest="width", default=224, help="Default width of images"
    )
    parser.add_option(
        "--height", dest="height", default=224, help="Default height of images"
    )
    parser.add_option(
        "--cat",
        dest="cat",
        default=f"{PATH}/data/NIBLES_data.csv",
        help="Catalog to get image names from.",
    )

    (options, args) = parser.parse_args()

    return options, args


def main():

    opt, arg = cmdline()

    # load the data
    df = pd.read_csv(opt.cat)

    width = opt.width
    height = opt.height

    # remove trailing slash in output path if it's there.
    opt.output = opt.output.rstrip("\/")

    # total number of images
    n_gals = df.shape[0]

    for row in df.itertuples():
        url = (
            "http://skyserver.sdss.org/dr14/SkyserverWS/ImgCutout/getjpeg"
            "?ra={}"
            "&dec={}"
            "&width={}"
            "&height={}".format(row.ra, row.dec, width, height)
        )
        if not os.path.isfile(f"{opt.output}/{row.nibles_id}.jpg"):
            try:
                img = skimage.io.imread(url)
                skimage.io.imsave(f"{opt.output}/{row.nibles_id}.jpg", img)
                time.sleep(0.03)
            except urllib.error.HTTPError:
                pass
        current = row.Index / n_gals * 100
        status = "{:.3f}% of {} completed.".format(current, n_gals)
        Printer(status)

    print("")


if __name__ == "__main__":
    main()

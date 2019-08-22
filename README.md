# ALFALFA-convnets

## Predicting gas mass fractions using ALFALFA x SDSS

A galaxy's cold gas content can determine its current and future star formation properties. Most of that cold gas in present-day galaxies is in the form of neutral atomic hydrogen, which radiates faintly through a 21 cm emission line. Since it is so difficult to observe this signal, many different heuristics have been developed in order to estimate the gas mass fraction (equivalent to a galaxy's gas mass normalized by its stellar mass). These proxies include, but are not limited to, simple [color-based methods](https://ui.adsabs.harvard.edu/abs/2004ApJ...611L..89K/abstract), [classical machine learning techniques](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.4509R/abstract), and [shallow neural networks](https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3796T/abstract). My work aims to improve existing approaches by also accounting for optical morphological information.

I use deep convolutional neural networks to process SDSS *gri* images (spanning 256 x 256 pixels, or roughly 100" x 100") of optical counterparts to ALFALFA detections in the local Universe (*cz* < 15,000 km/s). By using [data augmentation](https://ui.adsabs.harvard.edu/abs/2015MNRAS.450.1441D/abstract), a [one-cycle learning rate schedule](https://arxiv.org/abs/1803.09820), the [Rectified Adam optimizer](https://github.com/LiyuanLucasLiu/RAdam) ([paper](https://arxiv.org/abs/1908.03265)), and a [resnet-34](https://arxiv.org/abs/1512.03385) architecture ([+ bag of tricks](https://arxiv.org/abs/1812.01187)), **I can predict gas mass fractions to within 0.22 dex RMSE for the SDSS x α.40 data set**.

## Usage

Download this repository by running
```
git clone https://github.com/jwuphysics/alfalfa-convnets.git
cd alfalfa-convnets
```

## Dependencies

Pytorch `>=1.0` and Fastai `>=1.0` are required to run this code. They can be installed together using the Anaconda command

```
conda install -c pytorch -c fastai fastai
```

## Data

All data were queried from the [SDSS DR14 image cutout service](http://skyserver.sdss.org/dr14/en/help/docs/api.aspx#imgcutout) using a download script similar to the one in our [metallicity prediction deep convnet](https://github.com/jwuphysics/galaxy-cnns). Positions were taken from the ALFALFA [α.40 catalogs](http://egg.astro.cornell.edu/alfalfa/data/) ([Haynes et al. 2011](https://ui.adsabs.harvard.edu/abs/2011AJ....142..170H/abstract)).

## Acknowledgments

This work began during the [MIAPP Programme on Galaxy Evolution](http://www.munich-iapp.de/programmes-topical-workshops/2019/galaxy-evolution/daily-schedule/) and was inspired by conversations with [Mike Jones (IAA)](http://amiga.iaa.es/p/321-Michael-G-Jones.htm) and [Luke Leisman (Valpariso)](https://www.valpo.edu/physics-astronomy/about/faculty-and-staff/lukas-leisman/). The Fastai [course](https://course.fast.ai/) and [software](https://github.com/fastai/fastai) developed by Jeremy Howard et al. have been immensely useful for this work.
#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.signal import convolve, gaussian
from scipy.ndimage.filters import generic_filter
from scipy.stats import norm, beta
from contracts import contract
from skimage.util import view_as_blocks
import sys

__version__ = "0.10.0"
__title__ = "TMQIr"
__summary__ = "TMQI revised"
__uri__ = "https://github.com/dvolgyes/TMQI"

# for the Python reimplementation, original authors in 'upstream'
__author__ = "David Völgyes"
__email__ = "david.volgyes@ieee.org"

__derived__ = True   # Meaning: reimplementation with deviation
__upstream_license__ = "BSD-like"  # see the website for exact details
__upstream_uri__ = "https://ece.uwaterloo.ca/~z70wang/research/tmqi/"
__upstream_doi__ = "10.1109/TIP.2012.2221725"
__upstream_ref__ = ('H. Yeganeh and Z. Wang,' +
                    '"Objective Quality Assessment of Tone Mapped Images,"' +
                    'IEEE Transactions on Image Processing,' +
                    'vol. 22, no. 2, pp. 657-667, Feb. 2013.')


class Metric(object):

    def __init__(self, *args, **kwargs):
        self.name = "Undefined"
        self.descriptors = list()
        self.lists = tuple()
        self.maps = tuple()
        self.no_reference = False
        self.full_reference = False
        self.luminance = False
        self.RGB = False
        self.cache = dict()

    @contract(RGB='array[NxMx3](float)')
    def _RGBtoY(self, RGB):
        M = np.asarray([[0.2126, 0.7152, 0.0722], ])
        Y = np.dot(RGB.reshape(-1, 3), M.T)
        return Y.reshape(RGB.shape[0:2])


def img_read(link, gray=False, shape=None, dtype=None, keep=False):
    if os.path.exists(link):
        if dtype is None:
            img = imread(link)
            if gray and len(img.shape) > 2:
                img = skimage.color.rgb2hsv(img)[..., 2]
        else:
            W, H = shape
            img = np.fromfile(link, dtype=dtype)
            if gray:
                img = img.reshape(H, W)
            else:
                img = img.reshape(H, W, -1)
    else:
        tempfile = wget.download(link, bar=None)
        img = img_read(tempfile, gray, shape, dtype)
        if not keep:
            os.remove(tempfile)
    return img.astype(np.float)


class TMQI(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "TMQI"
        self.descriptors = ("Q", "S", "N")
        self.lists = ("s_local",)
        self.maps = ("s_local",)
        self.no_reference = False
        self.full_reference = True
        self.luminance = True
        self.RGB = False
        self.original = True

        if len(args) + len(kwargs) > 0:
            self.__call__(*args, **kwargs)

    @contract(hdrImage='array[NxMx3](float)|array[NxM](float),N>10,M>10',
              ldrImage='array[NxMx3](float)|array[NxM](float)')
    def __call__(self, hdrImage, ldrImage, window=None):
        # images must have same dimenions
        assert hdrImage.shape == ldrImage.shape

        if len(hdrImage.shape) == 3 and len(ldrImage.shape) == 3:
            # Processing RGB images
            L_hdr = self._RGBtoY(hdrImage)
            L_ldr = self._RGBtoY(ldrImage)
            return self._TMQI_gray(L_hdr, L_ldr, window)

        # input is already grayscale
        return self._TMQI_gray(hdrImage, ldrImage, window)

    @contract(hdrImage='array[NxM](float)',
              ldrImage='array[NxM](float)',
              window='None|array[UxV],U<N,V<M,U>=2,V>=2')
    def _TMQI_gray(self, hdrImage, ldrImage, window=None):
        a = 0.8012
        Alpha = 0.3046
        Beta = 0.7088
        lvl = 5  # levels
        weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

        M, N = hdrImage.shape

        if window is None:
            gauss = gaussian(11, 1.5)
            window = np.outer(gauss, gauss)

        # unnecessary, it is just for the sake of parallels with the matlab code
        L_hdr = hdrImage
        L_ldr = ldrImage

        # Naturalness should be calculated before rescaling
        N = self._StatisticalNaturalness(ldrImage)

        # The images should have the same dynamic ranges, e.g. [0,255]

        factor = float(2**32 - 1.)

        if self.original:
            L_hdr = factor * (L_hdr - L_hdr.min()) / (L_hdr.max() - L_hdr.min())
        else:
            # but we really should scale them similarly...
            L_hdr = factor * (L_hdr - L_hdr.min()) / (L_hdr.max() - L_hdr.min())
            L_ldr = factor * (L_ldr - L_ldr.min()) / (L_ldr.max() - L_ldr.min())

        S, s_local, s_maps = self._StructuralFidelity(L_hdr, L_ldr, lvl, weight, window)
        Q = a * (S ** Alpha) + (1. - a) * (N ** Beta)
        return Q, S, N, s_local, s_maps,

    @contract(L_hdr='array[NxM](float),N>0,M>0',
              L_ldr='array[NxM](float),N>0,M>0')
    def _StructuralFidelity(self, L_hdr, L_ldr, level, weight, window):

        f = 32
        s_local = []
        s_maps = []
        kernel = np.ones((2, 2)) / 4.0

        for _ in range(level):
            f = f / 2
            sl, sm = self._Slocal(L_hdr, L_ldr, window, f)

            s_local.append(sl)
            s_maps.append(sm)

            # averaging
            filtered_im1 = convolve(L_hdr, kernel, mode='valid')
            filtered_im2 = convolve(L_ldr, kernel, mode='valid')

            # downsampling
            L_hdr = filtered_im1[::2, ::2]
            L_ldr = filtered_im2[::2, ::2]

        S = np.prod(np.power(s_local, weight))
        return S, s_local, s_maps

    @staticmethod
    @contract(img1='array[NxM](float),N>0,M>0',
              img2='array[NxM](float),N>0,M>0',
              sf='float,>0')
    def _Slocal(img1, img2, window, sf, C1=0.01, C2=10.):

        window = window / window.sum()

        mu1 = convolve(window, img1, 'valid')
        mu2 = convolve(window, img2, 'valid')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve(img1 * img1, window, 'valid') - mu1_sq
        sigma2_sq = convolve(img2 * img2, window, 'valid') - mu2_sq

        sigma1 = np.sqrt(np.maximum(sigma1_sq, 0))
        sigma2 = np.sqrt(np.maximum(sigma2_sq, 0))

        sigma12 = convolve(img1 * img2, window, 'valid') - mu1_mu2

        CSF = 100.0 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(- (0.114 * sf) ** 1.1)
        u_hdr = 128 / (1.4 * CSF)
        sig_hdr = u_hdr / 3.

        sigma1p = norm.cdf(sigma1, loc=u_hdr, scale=sig_hdr)

        u_ldr = u_hdr
        sig_ldr = u_ldr / 3.

        sigma2p = norm.cdf(sigma2, loc=u_ldr, scale=sig_ldr)

        s_map = ((2 * sigma1p * sigma2p + C1) / (sigma1p**2 + sigma2p**2 + C1)
                 * ((sigma12 + C2) / (sigma1 * sigma2 + C2)))
        s = np.mean(s_map)
        return s, s_map

    @contract(L_ldr='array[NxM](float),N>0,M>0', win='int,>0')
    def _StatisticalNaturalness(self, L_ldr, win=11):
        phat1 = 4.4
        phat2 = 10.1
        muhat = 115.94
        sigmahat = 27.99
        u = np.mean(L_ldr)

        # moving window standard deviation using reflected image
        if self.original:
            W, H = L_ldr.shape
            w_extra = (11 - W % 11)
            h_extra = (11 - H % 11)
            # zero padding to simulate matlab's behaviour
            if w_extra > 0 or h_extra > 0:
                test = np.pad(L_ldr, pad_width=((0, w_extra), (0, h_extra)), mode='constant')
            else:
                test = L_ldr
            # block view with fixed block size, like in the original article
            view = view_as_blocks(test, block_shape=(11, 11))
            sig = np.mean(np.std(view, axis=(-1, -2)))
        else:
            # deviation: moving window with reflected borders
            sig = np.mean(generic_filter(L_ldr, np.std, size=win))

        beta_mode = (phat1 - 1.) / (phat1 + phat2 - 2.)
        C_0 = beta.pdf(beta_mode, phat1, phat2)
        C = beta.pdf(sig / 64.29, phat1, phat2)
        pc = C / C_0
        B = norm.pdf(u, muhat, sigmahat)
        B_0 = norm.pdf(muhat, muhat, sigmahat)
        pb = B / B_0
        N = pb * pc
        return N


class TMQIr(TMQI):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "TMQIrev"
        self.descriptors = ("Q", "S", "N")
        self.lists = ("s_local",)
        self.maps = ("s_local",)
        self.no_reference = False
        self.full_reference = True
        self.luminance = True
        self.RGB = False
        self.original = False


if __name__ == "__main__":
    if len(sys.argv) == 1:
        import doctest
        doctest.testmod()

    if len(sys.argv) > 1:  # there are command line parameters
        # these imports are unnecessary if the code is used as a library
        from optparse import OptionParser
        from scipy.misc import imsave
        from imageio import imread
        import os.path
        import wget
        import skimage.color

        usage = ("usage: %prog [options] HDR_image LDR_image\n" +
                 "The images could be files or a http(s)/ftp link.")
        parser = OptionParser(usage=usage)

        parser.add_option("-t", "--type",
                          type="string",
                          dest="maptype",
                          help="s_map file type (default: float32)",
                          default="float32")

        parser.add_option("-m", "--smap_file",
                          type="string",
                          dest="smap",
                          help="s_map file name prefix. (default: s_map_)",
                          default="s_map_")

        parser.add_option("-p", "--precision",
                          type="int",
                          dest="precision",
                          help="precision (number of decimals) (default: 4)",
                          default=4)

        parser.add_option("-W", "--width",
                          type="int",
                          dest="width",
                          help="image width (mandatory for RAW files)"
                          " (default: None)",
                          default=None)

        parser.add_option("-H", "--height",
                          type="int",
                          dest="height",
                          help="image height (mandatory for RAW files)"
                          " (default: None)",
                          default=None)

        parser.add_option("-i", "--input_type",
                          type="string",
                          dest="input",
                          help="type of the input images: float32/float64"
                          " for RAW images\n"
                          "None for regular images opening with scipy"
                          " (e.g. png) (default: None)",
                          default=None)

        parser.add_option("-g", "--gray",
                          dest="gray",
                          action="store_true",
                          help="gray input (ligthness/brightness)"
                          "  (default: RGB)",
                          default=False)

        parser.add_option("-Q", "--report-Q",
                          dest="report_q",
                          action="store_true",
                          help="report quality index",
                          default=True)

        parser.add_option("-S", "--report-S",
                          dest="report_s",
                          action="store_true",
                          help="report structural similarity",
                          default=False)

        parser.add_option("-L", "--report-SL",
                          dest="report_sl",
                          action="store_true",
                          help="report maps (S_locals)",
                          default=False)

        parser.add_option("-N", "--report-N",
                          dest="report_n",
                          action="store_true",
                          help="report naturalness",
                          default=False)

        parser.add_option("-M", "--report-MAPS",
                          dest="report_maps",
                          action="store_true",
                          help="report maps",
                          default=False)

        parser.add_option("-q", "--no-report-Q",
                          dest="report_q",
                          action="store_false",
                          help="do not report quality index")

        parser.add_option("-s", "--no-report-S",
                          dest="report_s",
                          action="store_false",
                          help="do not report structural similarity")

        parser.add_option("-l", "--no-report-SL",
                          dest="report_sl",
                          action="store_false",
                          help="do not report maps (S_locals)")

        parser.add_option("-n", "--no-report-N",
                          dest="report_n",
                          action="store_false",
                          help="do not report naturalness")

        parser.add_option("--quiet",
                          dest="quiet",
                          action="store_true",
                          help="suppress variable names in the report")

        parser.add_option("--verbose",
                          dest="quiet",
                          action="store_false",
                          help="use variable names in the report (default)",
                          default=False)

        parser.add_option("--keep",
                          dest="keep",
                          action="store_true",
                          help="keep downloaded files (default: False)",
                          default=False)

        parser.add_option("-r","--revised",
                          dest="revised",
                          action="store_true",
                          help="Enable revised TMQI. (default: Original)")

        (options, args) = parser.parse_args()

        if len(args) != 2:
            print("Exactly two input files are needed: HDR and LDR.")
            sys.exit(0)

        if options.input is not None:
            W, H = options.width, options.height
            shape = (W, H)
            dtype = np.dtype(options.input)
        else:
            shape = None
            dtype = None

        hdr = img_read(args[0], options.gray, shape, dtype, options.keep)
        ldr = img_read(args[1], options.gray, shape, dtype, options.keep)

        if options.revised:
            Q, S, N, s_local, s_maps = TMQIr()(hdr, ldr)
        else:
            Q, S, N, s_local, s_maps = TMQI()(hdr, ldr)

        prec = options.precision
        Q, S, N = np.round(Q, prec), np.round(S, prec), np.round(N, prec)
        s_local_str = " ".join(map(str, np.round(s_local, prec)))

        result = ""

        if options.report_q:
            if not options.quiet:
                result += "Q: "
            result += "%s " % Q

        if options.report_s:
            if not options.quiet:
                result += "S: "
            result += "%s " % S

        if options.report_n:
            if not options.quiet:
                result += "N: "
            result += "%s " % N

        if options.report_sl:
            if not options.quiet:
                result += "S_locals: "
            result += "%s " % s_local_str
        print(result.strip())

        if options.report_maps:
            for idx, sm in enumerate(s_maps):
                filename = "%s%i.%s" % (options.smap, idx + 1, options.maptype)

                try:
                    out = sm.astype(options.maptype)
                    out.tofile(filename)
                except TypeError:
                    imsave(filename, sm)

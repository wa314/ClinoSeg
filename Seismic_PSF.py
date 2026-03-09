# after: https://gist.github.com/kwinkunks/104a5474acfbc32d7f415d1f143de443

import numpy as np
import matplotlib.pyplot as plt
import pylops
from scipy.signal import convolve2d
from PIL import Image
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import bruges

class SeismicModel:
    def __init__(self, nx=61, nz=61, dx=4, dz=4, v0=2000, kv=0):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.v0 = v0
        self.kv = kv
        self.x = np.arange(nx) * dx
        self.z = np.arange(nz) * dz
        self.velocity_model = np.outer(np.ones(nx), v0 + kv * self.z)
        self.reflectivity_model = np.zeros((nx, nz))
        self.sources = self._initialize_sources(10)
        self.receivers = self._initialize_receivers(21)

    def _initialize_sources(self, ns):
        sx = np.linspace(self.dx * 10, (self.nx - 10) * self.dx, ns)
        sz = np.zeros(ns)
        return np.vstack((sx, sz))

    def _initialize_receivers(self, nr):
        rx = np.linspace(10 * self.dx, (self.nx - 10) * self.dx, nr)
        rz = np.zeros(nr)
        return np.vstack((rx, rz))

    def set_reflectivity(self, position):
        self.reflectivity_model[position] = 1

    def plot_reflectivity(self):
        fig, ax = plt.subplots(figsize=(15, 4))
        extent = [self.x[0], self.x[-1], self.z[-1], self.z[0]]
        im = ax.imshow(self.reflectivity_model.T, cmap='magma_r', extent=extent, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.scatter(self.receivers[0], self.receivers[1], marker='v')
        ax.scatter(self.sources[0], self.sources[1], marker='o')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Reflectivity')
        ax.set_ylim(self.z[-1], self.z[0] - 20)
        plt.show()

class SeismicSimulation:
    def __init__(self, model, nt=651, dt=0.004, f0=1):
        self.model = model
        self.nt = nt
        self.dt = dt
        self.t = np.arange(nt) * dt
        self.wav, self.wavt, self.wavc = pylops.utils.wavelets.ricker(self.t[:41], f0=f0)
        self.lsm = pylops.waveeqprocessing.LSM(
            model.z, model.x, self.t, model.sources, model.receivers, 
            model.v0, self.wav, self.wavc, mode='analytic'
        )

    def demigrate(self):
        d = self.lsm.Demop * self.model.reflectivity_model.ravel()
        return d.reshape(len(self.model.sources[0]), len(self.model.receivers[0]), self.nt)

    def migrate(self, data):
        madj = self.lsm.Demop.H * data.ravel()
        madj = madj.reshape(self.model.nx, self.model.nz)
        window = np.clip(np.blackman(61), 0, 1)
        window2d = np.sqrt(np.outer(window, window))
        return window2d * madj.T

    def compute_psf(self, refl):
        d = self.lsm.Demop * refl.ravel()
        return self.migrate(d)

    def plot_psf(self, psf):
        plt.imshow(psf)
        plt.show()

def make_rc(imp):
    """
    Compute reflection coefficients.
    """
    imp = np.pad(imp, pad_width=[(0, 1), (0, 0)], mode='edge')
    upper = imp[:-1, :]
    lower = imp[1:, :]
    return (lower - upper) / (lower + upper)

def load_image(path):
    return Image.open(path)

def cluster_image(image, eps=0.1, min_samples=25):
    im = np.array(image)[..., :3]
    X = np.reshape(im, (-1, 3))[::50]
    clu = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=6).fit(X)
    colours = []
    for label in np.unique(clu.labels_):
        if label != -1:
            sub = X[clu.labels_ == label]
            colours.append(np.mean(sub, axis=0))
    return colours

def generate_synthetic_seismic(colours, image, min_samples_threshold=25):
    im = np.array(image)[..., :3]
    kdtree = cKDTree(colours)
    _, ix = kdtree.query(im)
    print(ix)
    print("---------")
    print(ix.shape)

    
    unique_labels, counts = np.unique(ix, return_counts=True)
    valid_labels = unique_labels[counts >= min_samples_threshold]
    label_mapping = {label: idx for idx, label in enumerate(valid_labels)}
    rocks = np.random.uniform(low=5000000, high=8000000, size=len(valid_labels))
    
   
    imp = np.zeros_like(ix, dtype=float)
    for label in valid_labels:
        imp[ix == label] = rocks[label_mapping[label]]
    
    rc = make_rc(imp[::5])
    return imp, rc

def plot_synthetic_seismic(img, syn_1d, syn_psf):
    ma1 = np.percentile(np.abs(syn_1d), 99.9)
    ma2 = np.percentile(np.abs(syn_psf), 99.9)
    fig, axs = plt.subplot_mosaic("AA;AA;AA;BC;BC", figsize=(12, 9))
    axs['A'].imshow(img, aspect='auto')
    axs['A'].axis('off')
    axs['B'].imshow(syn_1d, aspect='auto', cmap='gray_r', vmin=-ma1, vmax=ma1)
    axs['B'].set_title('1D convolution with wavelet')
    axs['C'].imshow(syn_psf, aspect='auto', cmap='gray_r', vmin=-ma2, vmax=ma2)
    axs['C'].set_title('2D convolution with PSF')
    plt.show()

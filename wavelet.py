# -*- coding: utf-8 -*-
import torch
import numpy as np
from scipy.fftpack import hilbert as ht
import pywt
import cv2
from torchvision import transforms
grayscale = transforms.Grayscale(num_output_channels=1)

'''
img should be a numpy array
'''


def wavelet_wrapper(wavelet_type, img, img_size, modality=None):
    if wavelet_type == 'real':
        return wavelet_real(img,img_size)
    elif wavelet_type == 'quat':
        return wavelet_quat(img,img_size, modality)
    else:
        raise Exception



@torch.no_grad()
def wavelet_quat(image,image_size, modality):
    
    ########## IMAGE ###############
    #image = imread(image)
    # image = image/255.0
    image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    # image = cv2.resize(image, (256, 256))
    image = cv2.resize(image, (image_size*2, image_size*2))

    # print sizes
    # print("Image size:", image.shape)

    gl, gh, fl, fh = get_filters()

    if args.is_best_4:
        ele, all_ = qwt(image, gl, gh, fl, fh, only_low=False, quad="all")
        subbands = []
        for subband in all_:
            subband = subband[2:,:]
            subband = np.expand_dims(subband, axis=0)
            subbands.append(subband)
        train = np.concatenate(subbands, axis=0)
        #train = torch.from_numpy(train.astype(np.float32))
        a = []
        lst_to_iterate = args.best_4
        # if modality=="ct":
        #     lst_to_iterate = args.ct_best_4
        # if modality=="t1":
        #     lst_to_iterate = args.t1_best_4
        # elif modality =="t2":
        #     lst_to_iterate = args.t2_best_4
        # else:
        #     lst_to_iterate = args.best_4
        for wav_num in lst_to_iterate:
            a.append(train[wav_num])
        q0,q1,q2,q3 = quat_mag_and_phase(*a)
        train = np.stack([q0,q1,q2,q3], axis = 0)
        return train
    else:
        q0, q1, q2, q3 = qwt(
                        image,
                        gl, gh, fl, fh, 
                        only_low=args.wavelet_quat_type == "low", 
                        quad="all"
                    )
    #q0, q1, q2, q3 = quat_mag_and_phase(q0, q1, q2, q3)

    q0, q1, q2, q3 = q0[2:,:], q1[2:,:], q2[2:,:], q3[2:,:]

    q0 = q0.reshape(q0.shape[0], q0.shape[1], 1)
    q1 = q1.reshape(q1.shape[0], q1.shape[1], 1)
    q2 = q2.reshape(q2.shape[0], q2.shape[1], 1)
    q3 = q3.reshape(q3.shape[0], q3.shape[1], 1)

    image = np.concatenate((q0, q1, q2, q3), axis=2)

    ########## MASK ###############                             
    # mask = imread(mask, as_gray=True)
    # mask = cv2.resize(mask, (256, 256))
    # mask = (mask>0.5).astype('float32')

    # mask = np.expand_dims(mask, axis=0)
    image = image.transpose(2,0,1)
    
    #print("IMAGE-->", np.shape(image))
    #print("MASK-->", np.shape(mask))
    
    #image_tensor = torch.from_numpy(image.astype(np.float32))
    #mask_tensor = torch.from_numpy(mask.astype(np.float32))

    # return tensors
    return image#, mask_tensor


@torch.no_grad()
def wavelet_real(img, image_size):
    if img.shape[1] >1:
        img = grayscale(img).squeeze().numpy()
    img = cv2.resize(img, (image_size * 2 - 4, image_size * 2 - 4))
    ll, lh, hl, hh = wavelet_transformation(img)

    qs = np.stack((ll, lh, hl, hh), axis=2)

    amp = np.linalg.norm(qs, axis=2)

    φ_num = (ll * hl + lh * hh) * 2
    φ_den = (ll * ll + lh * lh - hl * hl - hh * hh)
    φ = np.arctan(φ_num / φ_den)
    φ = np.nan_to_num(φ)

    θ_num = (ll * lh + hl * hh)
    θ_den = (ll * ll - lh * lh + hl * hl - hh * hh)
    θ = np.arctan(θ_num / θ_den)
    θ = np.nan_to_num(θ)

    ψ = 0.5 * np.arctan(2 * (ll * hh - hh * lh))
    ψ = np.nan_to_num(ψ)

    # θ = (θ - np.min(θ)) / np.ptp(θ)
    # ψ = (ψ - np.min(ψ)) / np.ptp(ψ)
    # φ = (φ - np.min(φ)) / np.ptp(φ)

    ei = np.exp(φ)
    ej = np.exp(θ)
    ek = np.exp(ψ)

    # ei = (ei - np.min(ei))/np.ptp(ei)
    # ej = (ej - np.min(ej))/np.ptp(ej)
    # ek = (ek - np.min(ek))/np.ptp(ek)
    # amp = (amp - np.min(amp))/np.ptp(amp)

    train = np.stack((amp, ei, ej, ek), axis=2)

    train = (train - np.min(train)) / np.ptp(train)

    '''
    train = qs
    train =(train - np.min(train))/np.ptp(train)
    '''
    # train = np.reshape(train, (
    # train.shape[2], train.shape[0], train.shape[1]))  # array of float32 (4,256,256) valori [0,1]
    train = np.transpose(train, (2, 0, 1))

    return train



def wavelet_transformation(img):
    # Wavelet transform of image
    # titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']

    coeffs2 = pywt.dwt2(img, 'bior1.3')  # Biorthogonal wavelet
    LL, (LH, HL, HH) = coeffs2
    # fig = plt.figure(figsize=(12, 3))
    # for i, a in enumerate([LL, LH, HL, HH]):
    #         ax = fig.add_subplot(1, 4, i + 1)
    #         ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    #         ax.set_title(titles[i], fontsize=10)
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    # fig.tight_layout()
    # plt.show()

    return LL, LH, HL, HH


def pywt_coeffs():
    # coefficients from pywt db8 (http://wavelets.pybytes.com/wavelet/db8/)
    gl = [-0.00011747678400228192,
    0.0006754494059985568,
    -0.0003917403729959771,
    -0.00487035299301066,
    0.008746094047015655,
    0.013981027917015516,
    -0.04408825393106472,
    -0.01736930100202211,
    0.128747426620186,
    0.00047248457399797254,
    -0.2840155429624281,
    -0.015829105256023893,
    0.5853546836548691,
    0.6756307362980128,
    0.3128715909144659,
    0.05441584224308161
    ]

    gh = [-0.05441584224308161,
    0.3128715909144659,
    -0.6756307362980128,
    0.5853546836548691,
    0.015829105256023893,
    -0.2840155429624281,
    -0.00047248457399797254,
    0.128747426620186,
    0.01736930100202211,
    -0.04408825393106472,
    -0.013981027917015516,
    0.008746094047015655,
    0.00487035299301066,
    -0.0003917403729959771,
    -0.0006754494059985568,
    -0.00011747678400228192
    ]
    return np.asarray(gl), np.asarray(gh)

# Compute Hilbert transform of the filters G_L and G_H
def get_hilbert_filters(gl, gh):
    fl = ht(gl)
    fh = ht(gh)
    return fl, fh

def get_filters():
    gl, gh = pywt_coeffs()
    fl, fh = get_hilbert_filters(gl, gh)
    return gl, gh, fl, fh



def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx* and
    *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers + 0.5), the
    ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def as_column_vector( v):
    """Return *v* as a column vector with shape (N,1).
    """
    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    # (Shamelessly cribbed from scipy.)
    newsize = np.asanyarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _column_convolve( X, h):
    """Convolve the columns of *X* with *h* returning only the 'valid' section,
    i.e. those values unaffected by zero padding. Irrespective of the ftype of
    *h*, the output will have the dtype of *X* appropriately expanded to a
    floating point type if necessary.
    We assume that h is small and so direct convolution is the most efficient.
    """
    Xshape = np.asanyarray(X.shape)
    h = h.flatten().astype(X.dtype)
    h_size = h.shape[0]

    #     full_size = X.shape[0] + h_size - 1
    #     print("full size:", full_size)
    #     Xshape[0] = full_size

    out = np.zeros(Xshape, dtype=X.dtype)
    for idx in range(h_size):
        out += X * h[idx]
    
    outShape = Xshape.copy()
    outShape[0] = abs(X.shape[0] - h_size) + 1

    return _centered(out, outShape)



def colfilter(X, h):
    """Filter the columns of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each input sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of input samples, and Y.shape =
    X.shape + [1 0].
    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
    """

    # Interpret all inputs as arrays
#     X = asfarray(X)
    X = np.asarray(X)
    h = as_column_vector(h)

    r, c = X.shape
    m = h.shape[0]
    m2 = np.fix(m*0.5)

    # Symmetrically extend with repeat of end samples.
    # Use 'reflect' so r < m2 works OK.
    xe = reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)

    # Perform filtering on the columns of the extended matrix X(xe,:), keeping
    # only the 'valid' output samples, so Y is the same size as X if m is odd.
    Y = _column_convolve(X[xe,:], h)

    return Y

def qwt(image, gl, gh, fl, fh, only_low=True, quad=1):
    '''Compute the QWT. Just compute the low frequency coefficients.
    Return L_G L_G, L_F L_G, L_G L_F, L_F L_F.'''

    if only_low:
        t1 = colfilter(image, gl)
        t1 = downsample(t1)
        lglg = colfilter(t1, gl)
        t2 = colfilter(image, fl)
        t2 = downsample(t2)
        lflg = colfilter(t2, gl)
        t3 = colfilter(image, gl)
        t3 = downsample(t3)
        lglf = colfilter(t3, fl)
        t4 = colfilter(image, fl)
        t4 = downsample(t4)
        lflf = colfilter(t4, fl)
        # lglg, lflg, lglf, lflf = t1, t2, t3, t4
        # lglg, lflg, lglf, lflf = full_quat_downsample(lglg, lflg, lglf, lflf)

        return lglg, lflg, lglf, lflf
    else:
        if quad==1:
            t1 = colfilter(image, gl)
            lglg = colfilter(t1, gl)
            t2 = colfilter(image, gl)
            lghg = colfilter(t2, gh)
            t3 = colfilter(image, gh)
            hglg = colfilter(t3, gl)
            t4 = colfilter(image, gh)
            hghg = colfilter(t4, gh)

            return lglg, lghg, hglg, hghg

        elif quad==2:
            t1 = colfilter(image, fl)
            lflg = colfilter(t1, gl)
            t2 = colfilter(image, fl)
            lfhg = colfilter(t2, gh)
            t3 = colfilter(image, fh)
            hflg = colfilter(t3, gl)
            t4 = colfilter(image, fh)
            hfhg = colfilter(t4, gh)

            return lflg, lfhg, hflg, hfhg

        elif quad==3:
            t1 = colfilter(image, gl)
            lglf = colfilter(t1, fl)
            t2 = colfilter(image, gl)
            lghf = colfilter(t2, fh)
            t3 = colfilter(image, gh)
            hglf = colfilter(t3, fl)
            t4 = colfilter(image, gh)
            hghf = colfilter(t4, fh)

            return lglf, lghf, hglf, hghf
        
        elif quad==4:
            t1 = colfilter(image, fl)
            lflf = colfilter(t1, fl)
            t2 = colfilter(image, fl)
            lfhf = colfilter(t2, fh)
            t3 = colfilter(image, fh)
            hflf = colfilter(t3, fl)
            t4 = colfilter(image, fh)
            hfhf = colfilter(t4, fh)

            return lflf, lfhf, hflf, hfhf
        
        elif quad=="all":
            t1 = colfilter(image, gl)
            t1 = downsample(t1)
            lglg = colfilter(t1, gl)
            t2 = colfilter(image, gl)
            t2 = downsample(t2)
            lghg = colfilter(t2, gh)
            t3 = colfilter(image, gh)
            t3 = downsample(t3)
            hglg = colfilter(t3, gl)
            t4 = colfilter(image, gh)
            t4 = downsample(t4)
            hghg = colfilter(t4, gh)

            t1 = colfilter(image, fl)
            t1 = downsample(t1)
            lflg = colfilter(t1, gl)
            t2 = colfilter(image, fl)
            t2 = downsample(t2)
            lfhg = colfilter(t2, gh)
            t3 = colfilter(image, fh)
            t3 = downsample(t3)
            hflg = colfilter(t3, gl)
            t4 = colfilter(image, fh)
            t4 = downsample(t4)
            hfhg = colfilter(t4, gh)

            t1 = colfilter(image, gl)
            t1 = downsample(t1)
            lglf = colfilter(t1, fl)
            t2 = colfilter(image, gl)
            t2 = downsample(t2)
            lghf = colfilter(t2, fh)
            t3 = colfilter(image, gh)
            t3 = downsample(t3)
            hglf = colfilter(t3, fl)
            t4 = colfilter(image, gh)
            t4 = downsample(t4)
            hghf = colfilter(t4, fh)

            t1 = colfilter(image, fl)
            t1 = downsample(t1)
            lflf = colfilter(t1, fl)
            t2 = colfilter(image, fl)
            t2 = downsample(t2)
            lfhf = colfilter(t2, fh)
            t3 = colfilter(image, fh)
            t3 = downsample(t3)
            hflf = colfilter(t3, fl)
            t4 = colfilter(image, fh)
            t4 = downsample(t4)
            hfhf = colfilter(t4, fh)

            # Mean of components
            # ll = (lglg + lflg + lglf + lflf)/2
            # lh = (lghg + lfhg + lghf + lfhf)/2
            # hl = (hglg + hflg + hglf + hflf)/2
            # hh = (hghg + hfhg + hghf + hfhf)/2

            ll = (lflg + lglf + lflf)/3
            lh = (lfhg + lghf + lfhf)/3
            hl = (hflg + hglf + hflf)/3
            hh = (hfhg + hghf + hfhf)/3

            # return ll, lh, hl, hh
            return (ll, lh, hl, hh), (lglg, lghg, hglg, hghg, lflg, lfhg, hflg, hfhg, lglf, lghf, hglf, hghf, lflf, lfhf, hflf, hfhf)


def quat_mag_and_phase(q0, q1, q2, q3):
    '''Compute the magnitude and phase quaternion representation.'''
    q_mp = np.asarray([q0, q1, q2, q3])

    phi = np.arctan(2*(q0+q1*q3)/(q0**2+q1**2-q2**2-q3**2))
    theta = np.arctan((q0*q1+q2*q3)/(q0**2-q1**2+q2**2-q3**2))
    psi = 1/2*np.arctan(2*(q0*q3-q3*q1))

    phi = np.nan_to_num(phi)
    theta = np.nan_to_num(theta)
    psi = np.nan_to_num(psi)

    q0_mag = np.linalg.norm(q_mp, axis=0, ord=2)
    q1_phase = np.exp(phi)
    q2_phase = np.exp(theta)
    q3_phase = np.exp(psi)

    return q0_mag, q1_phase, q2_phase, q3_phase


def downsample(component):
    return component[::2, ::2]

def full_quat_downsample(q0, q1, q2, q3):
    q0 = downsample(q0)
    q1 = downsample(q1)
    q2 = downsample(q2)
    q3 = downsample(q3)
    return q0, q1, q2, q3


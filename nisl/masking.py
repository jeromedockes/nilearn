"""
Utilities to compute a brain mask from EPI images
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD
import warnings

import numpy as np
from scipy import ndimage

from . import utils

###############################################################################
# Operating on connect component
###############################################################################


def _largest_connected_component(mask):
    """
    Return the largest connected component of a 3D mask array.

    Parameters
    -----------
    mask: 3D boolean array
        3D array indicating a mask.

    Returns
    --------
    mask: 3D boolean array
        3D array indicating a mask, with only one connected component.
    """
    # We use asarray to be able to work with masked arrays.
    mask = np.asarray(mask)
    labels, label_nb = ndimage.label(mask)
    if not label_nb:
        raise ValueError('No non-zero values: no connected components')
    if label_nb == 1:
        return mask.astype(np.bool)
    label_count = np.bincount(labels.ravel())
    # discard 0 the 0 label
    label_count[0] = 0
    return labels == label_count.argmax()


def extrapolate_out_mask(data, mask, iterations=1):
    """ Extrapolate values outside of the mask.
    """
    if iterations > 1:
        data, mask = extrapolate_out_mask(data, mask,
                                          iterations=iterations - 1)
    new_mask = ndimage.binary_dilation(mask)
    larger_mask = np.zeros(np.array(mask.shape) + 2, dtype=np.bool)
    larger_mask[1:-1, 1:-1, 1:-1] = mask
    # Use nans as missing value: ugly
    masked_data = np.zeros(larger_mask.shape)
    masked_data[1:-1, 1:-1, 1:-1] = data.copy()
    masked_data[np.logical_not(larger_mask)] = np.nan
    outer_shell = larger_mask.copy()
    outer_shell[1:-1, 1:-1, 1:-1] = new_mask - mask
    outer_shell_x, outer_shell_y, outer_shell_z = np.where(outer_shell)
    extrapolation = list()
    for i, j, k in [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
                    (1, 0, 0), (-1, 0, 0)]:
        this_x = outer_shell_x + i
        this_y = outer_shell_y + j
        this_z = outer_shell_z + k
        extrapolation.append(masked_data[this_x, this_y, this_z])
        
    extrapolation = np.array(extrapolation)
    extrapolation = (np.nansum(extrapolation, axis=0)
                       / np.sum(np.isfinite(extrapolation), axis=0))
    extrapolation[np.logical_not(np.isfinite(extrapolation))] = 0
    new_data = np.zeros_like(masked_data)
    new_data[outer_shell] = extrapolation
    new_data[larger_mask] = masked_data[larger_mask]
    return new_data[1:-1, 1:-1, 1:-1], new_mask


###############################################################################
# Utilities to compute masks
###############################################################################


def compute_epi_mask(mean_epi, lower_cutoff=0.2, upper_cutoff=0.9,
                 connected=True, opening=True, exclude_zeros=False,
                 ensure_finite=True, verbose=0):
    """
    Compute a brain mask from fMRI data in 3D or 4D ndarrays.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    lower_cutoff and upper_cutoff of the total image histogram.

    In case of failure, it is usually advisable to increase lower_cutoff.

    Parameters
    ----------
    mean_epi: 3D ndarray
        EPI image, used to compute the mask.
    lower_cutoff : float, optional
        lower fraction of the histogram to be discarded.
    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.
    connected: boolean, optional
        if connected is True, only the largest connect component is kept.
    opening: boolean, optional
        if opening is True, an morphological opening is performed, to keep
        only large structures. This step is useful to remove parts of
        the skull that might have been included.
    ensure_finite: boolean
        If ensure_finite is True, the non-finite values (NaNs and infs)
        found in the images will be replaced by zeros
    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
    verbose: integer, optional

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask
    """
    if verbose > 0:
       print "EPI mask computation"
    if len(mean_epi.shape) == 4:
        mean_epi = mean_epi.mean(axis=-1)
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        mean_epi[np.logical_not(np.isfinite(mean_epi))] = 0
    sorted_input = np.sort(np.ravel(mean_epi))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = np.floor(lower_cutoff * len(sorted_input))
    upper_cutoff = np.floor(upper_cutoff * len(sorted_input))

    delta = sorted_input[lower_cutoff + 1:upper_cutoff + 1] \
            - sorted_input[lower_cutoff:upper_cutoff]
    ia = delta.argmax()
    threshold = 0.5 * (sorted_input[ia + lower_cutoff]
                        + sorted_input[ia + lower_cutoff + 1])

    mask = (mean_epi >= threshold)

    if connected:
        mask = _largest_connected_component(mask)
    if opening:
        mask = ndimage.binary_opening(mask.astype(np.int), iterations=2)
    return mask.astype(bool)

def intersect_masks(input_masks, threshold=0.5, connected=True):
    """
        Given a list of input mask images, generate the output image which
        is the the threshold-level intersection of the inputs

        Parameters
        ----------
        input_masks: list of strings or ndarrays
            paths of the input images nsubj set as len(input_mask_files), or
            individual masks.

        output_filename, string:
            Path of the output image, if None no file is saved.

        threshold: float within [0, 1[, optional
            gives the level of the intersection.
            threshold=1 corresponds to keeping the intersection of all
            masks, whereas threshold=0 is the union of all masks.
            
        connected: bool, optional
            If true, extract the main connected component

        Returns
        -------
            grp_mask, boolean array of shape the image shape
    """
    grp_mask = None
    if threshold > 1:
        raise ValueError('The threshold should be < 1')
    if threshold <0:
        raise ValueError('The threshold should be > 0')
    threshold = min(threshold, 1 - 1.e-7)

    for this_mask in input_masks:
        if grp_mask is None:
            grp_mask = this_mask.copy().astype(np.int)
        else:
            # If this_mask is floating point and grp_mask is integer, numpy 2
            # casting rules raise an error for in-place addition. Hence we do it
            # long-hand. XXX should the masks be coerced to int before addition?
            grp_mask = grp_mask + this_mask
    
    grp_mask = grp_mask > (threshold * len(list(input_masks)))
    
    if np.any(grp_mask > 0) and connected:
        grp_mask = _largest_connected_component(grp_mask)
    
    return grp_mask > 0


def compute_session_epi_mask(session_epi, lower_cutoff=0.2, upper_cutoff=0.9,
            connected=True, opening=True, threshold=0.5, exclude_zeros=False,
            return_mean=False, verbose=0):
    """ Compute a common mask for several sessions of fMRI data.

    Uses the mask-finding algorithmes to extract masks for each
    session, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.


    Parameters
    ----------
    session_files: list of list of strings
        A list of list of nifti filenames. Each inner list
        represents a session.

    threshold: float, optional
        the inter-session threshold: the fraction of the
        total number of session in for which a voxel must be in the
        mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.

    lower_cutoff: float, optional
        lower fraction of the histogram to be discarded.

    upper_cutoff: float, optional
        upper fraction of the histogram to be discarded.

    connected: boolean, optional
        if connected is True, only the largest connect component is kept.

    exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.

    Returns
    -------
    mask : 3D boolean ndarray
        The brain mask

    mean : 3D float array
        The mean image
    """
    masks = []
    for index, session in enumerate(session_epi):
        if utils.is_a_niimg(session):
            session = session.get_data()
        this_mask = compute_epi_mask(session,
                lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff,
                connected=connected, exclude_zeros=exclude_zeros)
        masks.append(this_mask.astype(np.int8))

    mask = intersect_masks(masks, connected=connected)

    return mask




###############################################################################
# Time series extraction
###############################################################################


def apply_mask(niimgs, mask_img, dtype=np.float32,
                     smooth=False, ensure_finite=True, transpose=False):
    """ Read the time series from the given sessions filenames, using the mask.

        Parameters
        -----------
        niimgs: list of 3D nifti file names, or 4D nifti filename.
            Files are grouped by session.
        mask: 3d ndarray
            3D mask array: true where a voxel should be used.
        smooth: False or float, optional
            If smooth is not False, it gives the size, in voxel of the
            spatial smoothing to apply to the signal.
        ensure_finite: boolean
            If ensure_finite is True, the non-finite values (NaNs and infs)
            found in the images will be replaced by zeros

        Returns
        --------
        session_series: ndarray
            3D array of time course: (session, voxel, time)
        header: header object
            The header of the first file.

        Notes
        -----
        When using smoothing, ensure_finite should be True: as elsewhere non
        finite values will spread accross the image.
    """
    mask = utils.check_niimg(mask_img)
    mask = mask_img.get_data().astype(np.bool)
    if smooth:
        # Convert from a sigma to a FWHM:
        smooth /= np.sqrt(8 * np.log(2))

    niimgs = utils.check_niimgs(niimgs)
    series = niimgs.get_data()
    affine = niimgs.get_affine()
    # We have 4D data
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        series[np.logical_not(np.isfinite(series))] = 0
    series = series.astype(dtype)
    affine = affine[:3, :3]
    # del data
    if isinstance(series, np.memmap):
        series = np.asarray(series).copy()
    if smooth:
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        smooth_sigma = smooth / vox_size
        for this_volume in np.rollaxis(series, -1):
            this_volume[...] = ndimage.gaussian_filter(this_volume,
                                                    smooth_sigma)
    series = series[mask]
    if transpose:
        series = series.T
    return series


def unmask(X, mask, transpose=False):
    """ Take masked data and bring them back into 3D

    This function is intelligent and will process data of any dimensions.
    It iterates until data has only one dimension and then it tries to
    unmask it. An error is raised if masked data has not the right number
    of voxels.

    Parameters
    ----------
   
    X: (list of)* numpy array
        Masked data. You can provide data of any dimension so if you want to
        unmask several images at one time, it is possible to give a list of
        images.
    mask: numpy array of boolean values
        Mask of the data
    """
    if mask.dtype != np.bool:
        warnings.warn('[unmask] Given mask had dtype %s.It has been converted'
            ' to bool.' % mask.dtype.name)
        mask = mask.astype(np.bool)

    if isinstance(X, np.ndarray) and len(X.shape) == 1:
        if X.shape[0] != mask.sum():
            raise ValueError('[unmask] Masked data and mask have not the same'
                ' number of voxels')
        img = np.zeros(mask.shape)
        img[mask] = X
        return img

    data = []
    if isinstance(X, np.ndarray):
        for x in X:
            img = unmask(x, mask)
            if transpose:
                data.append(img[..., np.newaxis])
            else:
                data.append(img[np.newaxis, ...])
        if transpose:
            data = np.concatenate(data, axis=-1)
        else:
            data = np.concatenate(data, axis=0)
    else:
        for x in X:
            img = unmask(x, mask)
            data.append(img)
    return data

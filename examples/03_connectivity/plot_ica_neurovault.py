"""
NeuroVault cross-study ICA maps.
================================

This example shows how to download statistical maps from
NeuroVault, label them with NeuroSynth terms,
and compute ICA components across all the maps.

See :func:`nilearn.datasets.neurovault.fetch_neurovault`
documentation for more details.

"""
# Author: Ben Cipollini
# License: BSD
# Ported from code authored by Chris Filo Gorgolewski, Gael Varoquaux
# https://github.com/NeuroVault/neurovault_analysis
import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA

from nilearn.datasets import neurovault as nv
from nilearn.image import new_img_like
from nilearn._utils import check_niimg


warnings.simplefilter('error', RuntimeWarning)  # Catch numeric issues in imgs
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)


def clean_img(img, dtype=np.float32):
    """ Remove nan/inf entries."""
    img = check_niimg(img)
    img_data = img.get_data().astype(dtype)
    img_data[np.isnan(img_data)] = 0
    img_data[np.isinf(img_data)] = 0
    return new_img_like(img, img_data)


######################################################################
# Get image and term data

# Download 500 images
print("Fetching Neurovault images; "
      "if you haven't downloaded any Neurovault data before "
      "this will take several minutes.")
nv_data = nv.fetch_neurovault(max_images=500, fetch_neurosynth_words=True)

images = nv_data['images']
term_weights = nv_data['word_frequencies']
vocabulary = nv_data['vocabulary']

# Clean and report term scores
term_weights[term_weights < 0] = 0
total_scores = np.mean(term_weights, axis=0)

print("\nTop 10 neurosynth terms from downloaded images:\n")
print('{2:>20}{0:>15} : {1:<15}\n{2:>20}{2:->30}'.format(
    'term', 'total score', ''))
for term_idx in np.argsort(total_scores)[-10:][::-1]:
    print('{0:>35} : {1:.3f}'.format(
        vocabulary[term_idx], total_scores[term_idx]))


######################################################################
# Reshape and mask images

from nilearn.datasets import load_mni152_brain_mask
from nilearn.input_data import NiftiMasker

print("\nReshaping and masking images.\n")

mask_img = load_mni152_brain_mask()
masker = NiftiMasker(mask_img=mask_img, memory='nilearn_cache')
masker = masker.fit()

# Images may fail to be transformed, and are of different shapes,
# so we need to transform one-by-one and keep track of failures.
X = []
is_usable = np.ones((len(images),), dtype=bool)
for index, image_path in enumerate(images):
    image = clean_img(image_path)
    try:
        X.append(masker.transform(image))
    except Exception as e:
        meta = nv_data['images_meta'][index]
        print("Failed to mask/reshape image: id: {0}; "
              "name: '{1}'; collection: {2}; error: {3}".format(
                  meta.get('id'), meta.get('name'),
                  meta.get('collection_id'), e))
        is_usable[index] = False

# Now reshape list into 2D matrix, and remove failed images from terms
X = np.vstack(X)
term_weights = term_weights[is_usable, :]


######################################################################
# Run ICA and map components to terms

print("Running ICA; may take time...")
n_components = 40
fast_ica = FastICA(n_components=n_components, random_state=0)
ica_maps = fast_ica.fit_transform(X.T).T

term_weights_for_components = np.dot(fast_ica.components_, term_weights)
print('Done, plotting results.')


######################################################################
# Generate figures

from nilearn import plotting

plotting.img_plotting.matplotlib.pyplot.rcParams[
    'figure.max_open_warning'] = n_components + 2

for index, (ic_map, ic_terms) in enumerate(zip(
        ica_maps, term_weights_for_components)):
    if -ic_map.min() > ic_map.max():
        # Flip the map's sign for prettiness
        ic_map = - ic_map
        ic_terms = - ic_terms

    ic_threshold = stats.scoreatpercentile(np.abs(ic_map), 90)
    ic_image = masker.inverse_transform(ic_map)
    display = plotting.plot_stat_map(
        ic_image, threshold=ic_threshold, colorbar=False, bg_img=mask_img)

    # Use the 4 terms weighted most as a title
    important_terms = vocabulary[np.argsort(ic_terms)[-4:]]
    title = '{0}: {1}'.format(index, ', '.join(important_terms[::-1]))
    display.title(title, size=16)

# Done.
plotting.show()

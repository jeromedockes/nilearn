"""
NeuroVault meta-analysis of stop-go paradigm studies.
=====================================================

This example shows how to download statistical maps from
NeuroVault

See :func:`nilearn.datasets.fetch_neurovault`
documentation for more details.

"""
import numpy as np
import scipy

from nilearn.datasets import neurovault as nv
from nilearn.image import new_img_like, load_img


######################################################################
# Fetch images for "successful stop minus go"-like protocols.

# Only 7 images match our critera; set max_images to 7
# so that if we already have them we won't look for more.
nv_data = nv.fetch_neurovault(
    max_images=7,
    cognitive_paradigm_cogatlas=nv.Contains('stop signal'),
    contrast_definition=nv.Contains('succ', 'stop', 'go'),
    map_type='T map')

images, collections = nv_data['images_meta'], nv_data['collections_meta']


######################################################################
# Display the paradigms and contrast definitions we've found.


def print_title(title):
    print('\n{0}\n{2:-<{1}}'.format(title, len(title), ''))


print_title("Paradigms we've downloaded:")
for im in images:
    print("{0:>10} : {1:<}".format(im['id'],
                                   im['cognitive_paradigm_cogatlas']))

print_title("Contrast definitions for downloaded images:")
for cd in np.unique([im['contrast_definition'] for im in images]):
    print("{0:>10}{1}".format("", cd))


######################################################################
# Visualize the data

from nilearn import plotting

print('\nPreparing plots for fetched images...')
for im in images:
    plotting.plot_glass_brain(im['absolute_path'],
                              title='image {0}'.format(im['id']))
print("Done")

######################################################################
# Compute statistics


from nilearn.image import mean_img


def t_to_z(t_scores, deg_of_freedom):
    p_values = scipy.stats.t.sf(t_scores, df=deg_of_freedom)
    z_values = scipy.stats.norm.isf(p_values)
    return z_values, p_values


# Compute z values
mean_maps = []
p_datas = []
z_datas = []
ids = set()
print("\nComputing maps...")
for collection in [col for col in collections
                   if not(col['id'] in ids or ids.add(col['id']))]:
    print_title("Collection {0}:".format(collection['id']))

    # convert t to z
    cur_imgs = [im for im in images if im['collection_id'] == collection['id']]
    image_z_niis = []
    for im in cur_imgs:
        # Load and validate the downloaded image.
        nii = load_img(im['absolute_path'])
        deg_of_freedom = im['number_of_subjects'] - 2
        print("{0:>10}Image {1:>4}: degrees of freedom: {2}".format(
            "", im['id'], deg_of_freedom))

        # Convert data, create new image.
        data_z, data_p = t_to_z(nii.get_data(), deg_of_freedom=deg_of_freedom)
        p_datas.append(data_p)
        z_datas.append(data_z)
        image_z_niis.append(nii)

    mean_map = mean_img(image_z_niis)
    plotting.plot_glass_brain(
        mean_map, title="Collection {0} mean map".format(collection['id']))
    mean_maps.append(mean_map)


# Fisher's z-score on all maps
def z_map(ref_img, z_data, affine):
    cut_coords = [-15, -8, 6, 30, 46, 62]
    z_meta_data = np.array(z_data).sum(axis=0) / np.sqrt(len(z_data))
    nii = new_img_like(ref_img, z_meta_data, affine)
    plotting.plot_stat_map(nii, display_mode='z', threshold=6,
                           cut_coords=cut_coords, vmax=12)


z_map(mean_maps[0], z_datas, mean_maps[0].get_affine())

# Fisher's z-score on combined maps
z_input_datas = [mean_nii.get_data() for mean_nii in mean_maps]
z_map(mean_maps[0], z_input_datas, mean_maps[0].get_affine())

plotting.show()

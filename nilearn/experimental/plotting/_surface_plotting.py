import numpy as np
from matplotlib import pyplot as plt

from nilearn import plotting as old_plotting


def plot_surf(img, parts=None, mesh=None, views=None, **kwargs):
    """Plot a SurfaceImage.

    TODO: docstring.
    """
    if mesh is None:
        mesh = img.mesh
    if parts is None:
        parts = list(img.data.parts.keys())
    if views is None:
        views=["lateral"]
    fig, axes = plt.subplots(
        len(views),
        len(parts),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(parts), 4),
    )
    axes = np.atleast_2d(axes)
    for view, ax_row in zip(views, axes):
        for ax, mesh_part in zip(ax_row, parts):
            old_plotting.plot_surf(
                mesh.parts[mesh_part],
                img.data.parts[mesh_part],
                hemi=mesh_part,
                view=view,
                axes=ax,
                title=mesh_part,
                **kwargs,
            )
    return fig

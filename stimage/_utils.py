import scipy.ndimage
import scipy as sp
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops
import os
import pandas as pd
from pathlib import Path
import warnings
from typing import Optional, Union
import cv2 as cv
import scanpy
from anndata import AnnData
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, threshold_multiotsu
from sklearn.metrics import jaccard_score

# from .utils import get_img_from_fig, checkType

from PIL import Image

# from .utils import get_img_from_fig, checkType
""" QC plot
"""


def QC_plot(
    adata: AnnData,
    library_id: str = None,
    name: str = None,
    data_alpha: float = 0.8,
    tissue_alpha: float = 1.0,
    cmap: str = "Spectral_r",
    spot_size: tuple = (5, 40),
    show_color_bar: bool = True,
    show_size_legend: bool = True,
    show_axis: bool = False,
    cropped: bool = True,
    image_scale: int = None,
    margin: int = 100,
    dpi: int = 150,
    output: str = None,
) -> Optional[AnnData]:
    """\
        QC plot for sptial transcriptomics data.

        Parameters
        ----------
        adata
            Annotated data matrix.
        library_id
            Library id stored in AnnData.
        data_alpha
            Opacity of the spot.
        tissue_alpha
            Opacity of the tissue.
        cmap
            Color map to use.
        spot_size
            Size of the spot (min, max).
        show_color_bar
            Show color bar or not.
        show_axis
            Show axis or not.
        show_size_legend
            Show size legend or not.
        name
            Name of the output figure file.
        output
            Save the figure as file or not.
        copy
            Return a copy instead of writing to adata.
        Returns
        -------
        Nothing
        """
    from sklearn.preprocessing import MinMaxScaler

    reads_per_spot = adata.to_df().sum(axis=1)
    scaler = MinMaxScaler(feature_range=spot_size)
    reads_per_spot_size = scaler.fit_transform(
        reads_per_spot.to_numpy().reshape(-1, 1))
    genes_per_spot = adata.to_df().astype(bool).sum(axis=1)

    scale_factor = 1
    if image_scale:
        scale_factor = 1 / image_scale

    imagecol = adata.obs["imagecol"] * scale_factor
    imagerow = adata.obs["imagerow"] * scale_factor
    # plt.rcParams['figure.dpi'] = dpi

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    vmin = min(genes_per_spot)
    vmax = max(genes_per_spot)
    # Plot scatter plot based on pixel of spots
    plot = a.scatter(
        imagecol,
        imagerow,
        edgecolor="none",
        alpha=data_alpha,
        s=reads_per_spot_size,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        cmap=plt.get_cmap(cmap),
        c=genes_per_spot,
    )

    if show_color_bar:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        axins = inset_axes(
            a,
            width="100%",
            height="100%",
            loc="upper left",
            bbox_to_anchor=(1.0, 0.73, 0.05, 0.35),
            bbox_transform=a.transAxes,
            borderpad=4.3,
        )
        cb = plt.colorbar(plot, cax=axins)
        cb.ax.set_xlabel("Number of Genes", fontsize=10)
        cb.ax.xaxis.set_label_coords(0.98, 1.20)
        cb.outline.set_visible(False)

    if show_size_legend:
        size_min, size_max = spot_size
        markers = [
            size_min,
            size_min + 1 / 3 * (size_max - size_min),
            size_min + 2 / 3 * (size_max - size_min),
            size_max,
        ]
        legend_markers = [plt.scatter([], [], s=i, c="grey") for i in markers]
        labels = [
            str(int(scaler.inverse_transform(np.array(i).reshape(1, 1))))
            for i in markers
        ]
        a.legend(
            handles=legend_markers,
            labels=labels,
            loc="lower left",
            bbox_to_anchor=(1, 0.05),
            scatterpoints=1,
            frameon=False,
            title="Number of Reads",
        )

    if not show_axis:
        a.axis("off")
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]]
    scale_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
    image_pil = Image.fromarray(image)
    image_pil.thumbnail(scale_size, Image.Resampling.LANCZOS)
    image = np.array(image_pil)
    # Overlay the tissue image
    a.imshow(
        image,
        alpha=tissue_alpha,
        zorder=-1,
    )

    if cropped:
        a.set_xlim(imagecol.min() - margin, imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin, imagerow.max() + margin)
        a.set_ylim(a.get_ylim()[::-1])
        # plt.gca().invert_yaxis()

    # fig.tight_layout()
    if output is not None:
        fig.savefig(
            output + "/" + name,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0)

    plt.show()


def gene_plot(
        adata: AnnData,
        method: str = "CumSum",
        slot: str = None,
        genes: Optional[Union[str, list]] = None,
        threshold: float = None,
        library_id: str = None,
        data_alpha: float = 1.0,
        tissue_alpha: float = 1.0,
        vmin: float = None,
        vmax: float = None,
        cmap: str = "Spectral_r",
        spot_size: Union[float, int] = 6.5,
        show_legend: bool = False,
        show_color_bar: bool = True,
        show_axis: bool = False,
        cropped: bool = True,
        image_scale: bool = True,
        margin: int = 100,
        name: str = None,
        output: str = None,
) -> Optional[FigureCanvasAgg]:
    """
    Gene expression plot for spatial transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    method : str, optional
        Use method to count. We provide: NaiveMean, NaiveSum, CumSum.
    slot : str, optional
        Slot name in adata.obsm.
    genes : str or list, optional
        Choose a gene or a list of genes.
    threshold : float, optional
        Threshold to filter genes.
    library_id : str, optional
        Library id stored in AnnData.
    data_alpha : float, optional
        Opacity of the spot.
    tissue_alpha : float, optional
        Opacity of the tissue.
    vmin : float, optional
        Minimum value for the color map.
    vmax : float, optional
        Maximum value for the color map.
    cmap : str, optional
        Color map to use.
    spot_size : float or int, optional
        Size of the spot.
    show_legend : bool, optional
        Show legend or not.
    show_color_bar : bool, optional
        Show color bar or not.
    show_axis : bool, optional
        Show axis or not.
    cropped : bool, optional
        Crop the image or not.
    image_scale : bool, optional
        Scale the image or not.
    margin : int, optional
        Margin size.
    name : str, optional
        Name of the output figure file.
    output : str, optional
        Save the figure as file or not.

    Returns
    -------
    Optional[FigureCanvasAgg]
        The generated gene expression plot as a FigureCanvasAgg object.
    """

    # plt.rcParams['figure.dpi'] = dpi

    if isinstance(genes, str):
        genes = [genes]
    colors = _gene_plot(adata, method, genes, slot)

    if threshold is not None:
        colors = colors[colors > threshold]

    index_filter = colors.index

    filter_obs = adata.obs.loc[index_filter]

    scale_factor = 1
    if image_scale:
        scale_factor = adata.uns['spatial'][library_id]['scalefactors'][
            'tissue_hires_scalef']

    imagecol = filter_obs["imagecol"] * scale_factor
    imagerow = filter_obs["imagerow"] * scale_factor

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots(figsize=(10, 10))
    if vmin:
        vmin = vmin
    else:
        vmin = min(colors)
    if vmax:
        vmax = vmax
    else:
        vmax = max(colors)
    # Plot scatter plot based on pixel of spots
    plot = a.scatter(
        imagecol,
        imagerow,
        edgecolor="none",
        alpha=data_alpha,
        s=spot_size,
        marker="h",
        vmin=vmin,
        vmax=vmax,
        cmap=plt.get_cmap(cmap),
        c=colors)

    if show_color_bar:
        cb = plt.colorbar(plot, cax=fig.add_axes(
            [0.9, 0.3, 0.03, 0.38]), cmap=cmap)
        cb.outline.set_visible(False)

    if not show_axis:
        a.axis('off')

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"]["hires"]
    # Overlay the tissue image
    a.imshow(image, alpha=tissue_alpha, zorder=-1, )

    if cropped:
        imagecol = adata.obs["imagecol"] * scale_factor
        imagerow = adata.obs["imagerow"] * scale_factor

        a.set_xlim(imagecol.min() - margin,
                   imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin,
                   imagerow.max() + margin)

        a.set_ylim(a.get_ylim()[::-1])

    if name is None:
        name = method
    if output is not None:
        fig.savefig(output + "/" + name, dpi=300,
                    bbox_inches='tight', pad_inches=0)

    return fig.canvas


def _gene_plot(adata, method, genes, slot):
    # Gene plot option

    if len(genes) == 0:
        raise ValueError('Genes shoule be provided, please input genes')

    elif len(genes) == 1:

        if genes[0] not in adata.var.index:
            raise ValueError(
                genes[0] + ' is not exist in the data, please try another gene')
        if slot is None:
            colors = adata[:, genes].to_df().iloc[:, -1]
        else:
            colors = adata.obsm[slot][genes[0]]

        return colors
    else:

        for gene in genes:
            if gene not in adata.var.index:
                genes.remove(gene)
                warnings.warn("We removed " + gene +
                              " because they not exist in the data")

            if len(genes) == 0:
                raise ValueError(
                    'All provided genes are not exist in the data')

        count_gene = adata[:, genes].to_df()

        if method is None:
            raise ValueError(
                'Please provide method to combine genes by NaiveMean/NaiveSum/CumSum')

        if method == "NaiveMean":
            present_genes = (count_gene > 0).sum(axis=1) / len(genes)

            count_gene = (count_gene.mean(axis=1)) * present_genes
        elif method == "NaiveSum":
            present_genes = (count_gene > 0).sum(axis=1) / len(genes)

            count_gene = (count_gene.sum(axis=1)) * present_genes

        elif method == "CumSum":
            count_gene = count_gene.cumsum(axis=1).iloc[:, -1]

        colors = count_gene

        return colors


def pattern_matrix(adata: AnnData,
                   genes: str = None,
                   library_id: str = None,
                   gc_slot: str = None,
                   pred_slot: str = None
                   ) -> Optional[np.float64]:
    """\
    Gene expression plot for spatial transcriptomics data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : str, optional
        Choose a gene or a list of genes.
    library_id : str, optional
        Library id stored in AnnData.
    gc_slot : str, optional
        Slot name for the gene counts.
    pred_slot : str, optional
        Slot name for the predicted values.

    Returns
    -------
    Optional[np.float64]
        IoU Score
    """

    # plt.rcParams['figure.dpi'] = dpi
    canvas = gene_plot(adata, genes=genes, slot=pred_slot, image_scale=True,
                       library_id=library_id, spot_size=115, tissue_alpha=0,
                       show_color_bar=False, show_axis=False)
    canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(
        canvas.tostring_rgb(),
        dtype='uint8')  # (H * W * 3,)

    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image_pred = image_flat.reshape(*reversed(canvas.get_width_height()), 3)

    canvas = gene_plot(adata, genes=genes, slot=gc_slot, image_scale=True,
                       library_id=library_id, spot_size=115, tissue_alpha=0,
                       show_color_bar=False, show_axis=False)

    canvas.draw()  # Draw the canvas, cache the renderer
    image_flat = np.frombuffer(
        canvas.tostring_rgb(),
        dtype='uint8')  # (H * W * 3,)

    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image_gc = image_flat.reshape(*reversed(canvas.get_width_height()), 3)

    image_pred_r = _generate_binary_img(image_pred)
    image_gc_r = _generate_binary_img(image_gc)

    score = jaccard_score(image_gc_r, image_pred_r, average=None)[0]

    return score


def _generate_binary_img(img: np.ndarray,
                         sigma: list = [1.0, 1.0],
                         ) -> np.ndarray:
    """\
    Generate binary image from input image.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    sigma : list, optional
        Standard deviation for Gaussian smoothing in X and Y directions, default is [1.0, 1.0].

    Returns
    -------
    np.ndarray
        Binary image.
    """

    thresh_b = threshold_multiotsu(img[:, :, 0])
    img_b_b = np.where((img[:, :, 0] < thresh_b[0]), True, False)
    thresh_r = threshold_multiotsu(img[:, :, 2])
    img_r_b = np.where((img[:, :, 2] < thresh_r[0]), True, False)

    img_r_b_f = sp.ndimage.filters.gaussian_filter(
        img_r_b, sigma, mode='constant')

    img_r = img_r_b_f[img[:, :, 0] != 255]

    return img_r


"""Reading and Writing
"""


def Read10X(
        path: Union[str, Path],
        genome: Optional[str] = None,
        count_file: str = "filtered_feature_bc_matrix.h5",
        library_id: str = None,
        load_images: Optional[bool] = True,
        source_image_path: Union[str, Path, None] = None,
        quality: str = "hires"
) -> AnnData:
    """\
    Read Visium data from 10X (wrap read_visium from scanpy)

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    load_images
        Load image or not.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']


    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """

    from scanpy import read_visium
    adata = read_visium(path, genome=None,
                        count_file=count_file,
                        library_id=library_id,
                        load_images=load_images,
                        source_image_path=source_image_path)
    adata.var_names_make_unique()
    adata.obs['sum_counts'] = np.array(adata.X.sum(axis=1))
    adata.obsm["spatial"] = adata.obsm["spatial"].astype(np.int)
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        adata.uns["spatial"][library_id]["images"][quality] = plt.imread(
            source_image_path, 0)
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality

    return adata


Image.MAX_IMAGE_PIXELS = None


def add_image(
    adata: AnnData,
    imgpath: Union[Path, str],
    library_id: str,
    quality: str = "hires",
    scale: float = 1.0,
    visium: bool = False,
    spot_diameter_fullres: float = 50,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Adding image data to the Anndata object

    Parameters
    ----------
    adata
        Annotated data matrix.
    imgpath
        Image path.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow'].
    visium
        Is this anndata read from Visium platform or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tissue_img** : `adata.uns` field
        Array format of image, saving by Pillow package.
    """

    if imgpath is not None and os.path.isfile(imgpath):
        try:
            img = plt.imread(imgpath, 0)

            if visium:
                adata.uns["spatial"][library_id]["images"][quality] = img
            else:
                adata.uns["spatial"] = {}
                adata.uns["spatial"][library_id] = {}
                adata.uns["spatial"][library_id]["images"] = {}
                adata.uns["spatial"][library_id]["images"][quality] = img
                adata.uns["spatial"][library_id]["use_quality"] = quality
                adata.uns["spatial"][library_id]["scalefactors"] = {}
                adata.uns["spatial"][library_id]["scalefactors"][
                    "tissue_" + quality + "_scalef"
                ] = scale
                adata.uns["spatial"][library_id]["scalefactors"][
                    "spot_diameter_fullres"
                ] = spot_diameter_fullres
                adata.obsm["spatial"] = adata.obs[[
                    "imagecol", "imagerow"]].values
                adata.obs[["imagecol", "imagerow"]
                          ] = adata.obsm["spatial"] * scale

            print("Added tissue image to the object!")

            return adata if copy else None
        except BaseException:
            raise ValueError(
                f"""\
            {imgpath!r} does not end on a valid extension.
            """
            )
    else:
        raise ValueError(
            f"""\
        {imgpath!r} does not end on a valid extension.
        """
        )
    return adata if copy else None


def ReadOldST(
        count_matrix_file: Union[str, Path] = None,
        spatial_file: Union[str, Path] = None,
        image_file: Union[str, Path] = None,
        library_id: str = "OldST",
        scale: float = 1.0,
        quality: str = "hires",
        spot_diameter_fullres: float = 50,
) -> AnnData:
    """\
    Read Old Spatial Transcriptomics data
    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to spatial location file.
    image_file
        Path to the tissue image file
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution
    Returns
    -------
    AnnData
    """

    adata = scanpy.read_text(count_matrix_file)
    spot_df = pd.read_csv(spot_path, index_col=0)
    comm_index = pd.Series(
        list(set(spot_df.index).intersection(set(adata.obs_names))))
    adata = adata[comm_index]
    adata.obs["imagecol"] = spot_df["X"]
    adata.obs["imagerow"] = spot_df["Y"]
    add_image(
        adata,
        library_id=library_id,
        quality=quality,
        imgpath=image_file,
        scale=scale,
        spot_diameter_fullres=spot_diameter_fullres,
    )

    return adata


# Test progress bar


def ensembl_to_id(
        adata: AnnData,
        ens_path: Union[Path, str] = None,
        verbose: bool = False,
        copy: bool = True,
) -> Optional[AnnData]:
    if ens_path:
        ens_path = ens_path
    else:
        ens_path = Path(__file__).parent.absolute() / "ensembl.tsv"
    ens_df = pd.read_csv(ens_path, sep="\t")
    a = pd.Index(ens_df["Ensembl ID(supplied by Ensembl)"]
                 ).intersection(adata.var_names)
    var_dic = dict(
        zip(ens_df["Ensembl ID(supplied by Ensembl)"], ens_df["Approved symbol"]))
    adata = adata[:, a].copy()
    adata.var_names = adata.var_names.map(var_dic)
    return adata if copy else None


# Test progress bar


def tiling(
        adata: AnnData,
        out_path: Union[Path, str] = "./tiling",
        library_id: str = None,
        crop_size: int = 40,
        target_size: int = 299,
        stain_normaliser=None,
        image_select: Union[str, np.ndarray] = "HE",
        verbose: bool = False,
        copy: bool = False,
        save_name: str = "tile_path",
) -> Optional[AnnData]:
    """\
    Tiling H&E images to small tiles based on spot spatial location

    Parameters
    ----------
    adata
        Annotated data matrix.
    out_path
        Path to save spot image tiles
    library_id
        Library id stored in AnnData.
    crop_size
        Size of tiles
    verbose
        Verbose output
    copy
        Return a copy instead of writing to adata.
    target_size
        Input size for convolutional neuron network
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tile_path** : `adata.obs` field
        Saved path for each spot image tiles
    """

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    # Check the exist of out_path
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if isinstance(image_select, str):
        image = adata.uns["spatial"][library_id]["images"][
            adata.uns["spatial"][library_id]["use_quality"]]
    else:
        image = image_select
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)

    if stain_normaliser:
        stain_normaliser.fit_source(scale_img(img_pillow))
    tile_names = []

    with tqdm(
            total=len(adata),
            desc="Tiling image",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for imagerow, imagecol in zip(
                adata.obs["imagerow"],
                adata.obs["imagecol"]):

            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )
            if stain_normaliser:
                tile = stain_normaliser.transform_tile(tile)
            # tile.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            tile = tile.resize((target_size, target_size))
            tile_name = library_id + "-" + \
                str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            out_tile = Path(out_path) / (tile_name + ".jpeg")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)
                    )
                )
            if tile.mode == "RGBA":
                tile = tile.convert('RGB')
            tile.save(out_tile, "JPEG")

            pbar.update(1)

    adata.obs[save_name] = tile_names
    return adata if copy else None


def calculate_bg(
        adata: AnnData,
        stain_normaliser=None,
        crop_size: int = 40,
        library_id: str = None,
        temp_TILE_PATH: Union[Path, str] = Path("./tmp"),
        copy: bool = False,
) -> Optional[AnnData]:
    """\
    Calculate the background pixels for each tiles

    Parameters
    ----------
    adata
        Annotated data matrix.
    stain_normaliser
        normaliser class
    library_id
        Library id stored in AnnData.
    crop_size
        Size of tiles
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tissue_area** : `adata.obs` field
        Calculate the background pixels for each tiles
    """

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    img = Image.fromarray(adata.uns["spatial"][library_id]['images']["fulres"])
    img_small = scale_img(img)
    if stain_normaliser:
        stain_normaliser.fit_source(scale_img(img_small))
        img_small = stain_normaliser.transform_tile(img_small)

    target_img_norm_filtered = filter_green(img_small, g_thresh=250)
    target_img_norm_filtered = filter_grays(
        target_img_norm_filtered, tolerance=3)
    tissue_mask = tissue_mask_grabcut(np.array(target_img_norm_filtered))
    tissue_mask_up_scale = tissue_mask.resize(img.size, Image.Resampling.LANCZOS)

    _TILE_PATH = temp_TILE_PATH / \
        (list(adata.uns["spatial"].keys())[0] + "_tissue_mask")
    _TILE_PATH.mkdir(parents=True, exist_ok=True)

    tiling(adata, _TILE_PATH, crop_size=crop_size, image_select=np.array(
        tissue_mask_up_scale), save_name="tile_tissue_mask_path")

    tissue_area_list = []
    for img_path in adata.obs["tile_tissue_mask_path"]:
        tile_mask = plt.imread(img_path, 0)
        tissue_area = (tile_mask > 200).sum() / tile_mask.size
        tissue_area_list.append(tissue_area)
    adata.obs["tissue_area"] = np.array(tissue_area_list)
    return adata if copy else None


# from .utils import get_img_from_fig, checkType


def tissue_area_plot(
        adata: AnnData,
        threshold: float = None,
        library_id: str = None,
        data_alpha: float = 1.0,
        tissue_alpha: float = 1.0,
        vmin: float = None,
        vmax: float = None,
        cmap: str = "Spectral_r",
        spot_size: Union[float, int] = 6.5,
        show_legend: bool = False,
        show_color_bar: bool = True,
        show_axis: bool = False,
        cropped: bool = True,
        margin: int = 100,
        name: str = None,
        output: str = None,
        copy: bool = False,
) -> Optional[AnnData]:
    """\
    Calculate the background pixels for each tiles

    Parameters
    ----------
    adata
        Annotated data matrix.
    threshold
        threshold to filter out tiles that less tissue area pixels
    library_id
        Library id stored in AnnData.
    data_alpha
        Opacity of the spot.
    tissue_alpha
        Opacity of the tissue.
    vmin
        Min value for color
    vmax
        Max value for color
    cmap
        Color map to use.
    spot_size
        Size of the spot.
    show_legend
        Show legend or not.
    show_color_bar
        Show color bar or not.
    show_axis
        Show axis or not.
    cropped
        Show Cropped image or not
    margin
        Margin to image edge.
    name
        Name of the output figure file.
    output
        Save the figure as file or not.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tissue_area** : `adata.obs` field
        Calculate the background pixels for each tiles
    """
    colors = adata.obs["tissue_area"]

    if threshold is not None:
        colors = colors[colors > threshold]

    index_filter = colors.index

    filter_obs = adata.obs.loc[index_filter]

    imagecol = filter_obs["imagecol"]
    imagerow = filter_obs["imagerow"]

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()
    if vmin:
        vmin = vmin
    else:
        vmin = min(colors)
    if vmax:
        vmax = vmax
    else:
        vmax = max(colors)
    # Plot scatter plot based on pixel of spots
    plot = a.scatter(
        imagecol,
        imagerow,
        edgecolor="none",
        alpha=data_alpha,
        s=spot_size,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        cmap=plt.get_cmap(cmap),
        c=colors)
    plot.set_clim(vmin=vmin, vmax=vmax)
    if show_color_bar:
        cb = plt.colorbar(plot, cax=fig.add_axes(
            [0.9, 0.3, 0.03, 0.38]), cmap=cmap)
        cb.outline.set_visible(False)

    if not show_axis:
        a.axis('off')

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]]
    # Overlay the tissue image
    a.imshow(image, alpha=tissue_alpha, zorder=-1, )

    if cropped:
        imagecol = adata.obs["imagecol"]
        imagerow = adata.obs["imagerow"]

        a.set_xlim(imagecol.min() - margin,
                   imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin,
                   imagerow.max() + margin)

        a.set_ylim(a.get_ylim()[::-1])

    if name is None:
        name = "tissue_area_plot"
    if output is not None:
        fig.savefig(output + "/" + name, dpi=plt.figure().dpi,
                    bbox_inches='tight', pad_inches=0)

    plt.show()


def thumbnail(img, size=(1000, 1000)):
    """Converts Pillow images to a different size without modifying the original image
    """
    img_thumbnail = img.copy()
    img_thumbnail.thumbnail(size)
    return img_thumbnail


def scale_img(img, scale_f=10):
    """Scales Pillow images to a different size
    """
    return img.resize((img.size[0] // scale_f, img.size[1] // scale_f))


def tissue_mask_grabcut(img):
    """Masks the tissue region using cv2
    """
    img_cv = img[:, :, ::-1]  # Convert RGB to BGR
    mask_initial = (np.array(Image.fromarray(img).convert('L'))
                    < 250).astype(np.uint8)

    # Grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv.grabCut(img_cv, mask_initial, None, bgdModel,
               fgdModel, 5, cv.GC_INIT_WITH_MASK)
    mask_final = np.where((mask_initial == 2) | (
        mask_initial == 0), 0, 1).astype('uint8')

    # Generate a rough 'filled in' mask of the tissue
    kernal_64 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (64, 64))
    mask_closed = cv.morphologyEx(mask_final, cv.MORPH_CLOSE, kernal_64)
    mask_opened = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernal_64)

    # Use rough mask to remove small debris in grabcut mask
    mask_cleaned = cv.bitwise_and(mask_final, mask_final, mask=mask_opened)
    mask_cleaned_pil = Image.fromarray(mask_cleaned.astype(bool))
    return mask_cleaned_pil


def filter_green(img, g_thresh=240):
    """Replaces green pixels greater than threshold with white pixels

    Used to remove background from tissue images
    """
    img = img.convert('RGB')
    r, g, b = img.split()
    green_mask = (np.array(g) > 240) * 255
    green_mask_img = Image.fromarray(green_mask.astype(np.uint8), 'L')
    white_image = Image.new('RGB', img.size, (255, 255, 255))
    img_filtered = img.copy()
    img_filtered.paste(white_image, mask=green_mask_img)
    return img_filtered


def filter_grays(img, tolerance=3):
    """Replaces gray pixels greater than threshold with white pixels

    Used to remove background from tissue images
    """
    img = img.convert('RGB')
    r, g, b = img.split()
    rg_diff = np.array(ImageChops.difference(r, g)) <= tolerance
    rb_diff = np.array(ImageChops.difference(r, b)) <= tolerance
    gb_diff = np.array(ImageChops.difference(g, b)) <= tolerance
    grays = (rg_diff & rb_diff & gb_diff) * 255
    grays_mask = Image.fromarray(grays.astype(np.uint8), 'L')
    white_image = Image.new('RGB', img.size, (255, 255, 255))
    img_filtered = img.copy()
    img_filtered.paste(white_image, mask=grays_mask)
    return img_filtered


def classification_preprocessing(anndata):
    gene_exp = anndata.to_df()
    gene_exp['library_id'] = anndata.obs['library_id']
    gene_exp_zscore = gene_exp.groupby('library_id')[list(
        gene_exp.iloc[:, :-1].columns)].apply(lambda x: (x - x.mean()) / (x.std()))
    anndata.obsm["true_gene_expression"] = pd.DataFrame(
        gene_exp_zscore.apply(lambda x: [0 if y <= 0 else 1 for y in x]),
        columns=anndata.to_df().columns, index=anndata.obs.index)

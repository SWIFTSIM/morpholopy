import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import unyt
from swiftsimio import load, mask
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.projection import project_pixel_grid
from swiftsimio.visualisation.slice import kernel_gamma
from swiftsimio.visualisation.smoothing_length_generation import (
    generate_smoothing_lengths,
)
from swiftsimio import swift_cosmology_to_astropy
from astropy.visualization import make_lupton_rgb


cmRdBu = plt.get_cmap("RdYlBu_r")
Sigma_min = 0.0
Sigma_max = 99.0
H2_over_HI_min = 10.0 ** (-1.9)
H2_over_HI_max = 10.0 ** (+2.4)

twelve_plus_logOH_solar = 8.69
twelve_plus_logOH_min = twelve_plus_logOH_solar - 0.5
twelve_plus_logOH_max = twelve_plus_logOH_solar + 0.5
Zsun = 0.0134
twelve_plus_log_OH_solar = 8.69

#########################
# nr of map plots
#########################

nr_text_boxes = 1
nr_map_plots = 6
nr_scatter_plots = 1
nr_total_plots = nr_text_boxes + nr_map_plots + nr_scatter_plots
#################################
# Miscellaneous
#################################

npix = int(512 / 2)

vmin = -0.98
vmax = 2.98

r_img_kpc = 30.0 * unyt.kpc
lbar_kpc = 15.0 * unyt.kpc
ypos_bar = 20.0 * unyt.kpc

size = 2.0 * r_img_kpc
radialbinsizes = [0.22, 0.8, 1.8]  # in kpc
pixsize_kpc = r_img_kpc.value / npix

bingrid = 0.8 * unyt.kpc
npix_coarse = int(2.0 * r_img_kpc / bingrid)

cmap = plt.get_cmap("Greens")
usergreen = cmap(0.7)
cmap = plt.get_cmap("Blues")
userblue = cmap(0.7)


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    try:
        if exponent is None:
            exponent = int(np.floor(np.log10(np.abs(num))))
        coeff = np.round(num / float(10 ** exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits

        return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)

    except:
        return "0"


def get_stars_surface_brightness_map(
    catalogue,
    halo_id,
    snapshot_filename,
    size,
    npix,
    r_img_kpc,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
):
    # center of the halo
    x = catalogue.positions.xcmbp[halo_id]
    y = catalogue.positions.ycmbp[halo_id]
    z = catalogue.positions.zcmbp[halo_id]

    # angular momentum of the stars (for projection)
    lx = catalogue.angular_momentum.lx_star[halo_id]
    ly = catalogue.angular_momentum.ly_star[halo_id]
    lz = catalogue.angular_momentum.lz_star[halo_id]

    angular_momentum_vector = np.array([lx.value, ly.value, lz.value])
    angular_momentum_vector /= np.linalg.norm(angular_momentum_vector)
    # needs to be in comoving coordinates for the mask
    region = [
        [
            x / catalogue.scale_factor - size / catalogue.scale_factor,
            x / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
        [
            y / catalogue.scale_factor - size / catalogue.scale_factor,
            y / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
        [
            z / catalogue.scale_factor - size / catalogue.scale_factor,
            z / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
    ]

    visualise_region = [x - 0.5 * size, x + 0.5 * size, y - 0.5 * size, y + 0.5 * size]

    data_mask = mask(snapshot_filename)
    data_mask.constrain_spatial(region)
    data = load(snapshot_filename, mask=data_mask)
    data.stars.coordinates = data.stars.coordinates.to_physical()
    data.stars.smoothing_lengths = generate_smoothing_lengths(
        coordinates=data.stars.coordinates,
        boxsize=data.metadata.boxsize,
        kernel_gamma=kernel_gamma,
        neighbours=11,
        speedup_fac=1,
        dimension=3,
    )

    luminosities = [
        data.stars.luminosities.GAMA_i,
        data.stars.luminosities.GAMA_r,
        data.stars.luminosities.GAMA_g,
    ]
    rgb_image_face = np.zeros((npix, npix, len(luminosities)))

    for ilum in range(len(luminosities)):
        # Face on projection
        data.stars.usermass = luminosities[ilum]
        pixel_grid = project_pixel_grid(
            data.stars,
            resolution=npix,
            project="usermass",
            parallel=True,
            region=visualise_region,
            rotation_center=unyt.unyt_array([x, y, z]),
            rotation_matrix=face_on_rotation_matrix,
            boxsize=data.metadata.boxsize,
            backend="subsampled",
        )
        x_range = visualise_region[1] - visualise_region[0]
        y_range = visualise_region[3] - visualise_region[2]
        units = 1.0 / (x_range * y_range)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))

        mass_map_face = unyt.unyt_array(pixel_grid, units=units)
        mass_map_face.convert_to_units(1.0 / unyt.pc ** 2)
        try:
            mass_map_face[mass_map_face == 0.0] = mass_map_face[
                mass_map_face > 0.0
            ].min()
        except:
            mass_map_face[mass_map_face == 0.0] = 1.0e-10

        rgb_image_face[:, :, ilum] = mass_map_face.T
    image_face = make_lupton_rgb(
        rgb_image_face[:, :, 0],
        rgb_image_face[:, :, 1],
        rgb_image_face[:, :, 2],
        Q=10,
        stretch=0.5,
    )
    # mask with circle
    lx, ly = mass_map_face.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    mask_circle = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    image_face[mask_circle, :] = 255
    H_kpc_gri = np.zeros(len(luminosities))
    rgb_image_edge = np.zeros((npix, npix, len(luminosities)))
    for ilum in range(len(luminosities)):
        # Face on projection
        data.stars.usermass = luminosities[ilum]
        pixel_grid = project_pixel_grid(
            data.stars,
            resolution=int(npix),
            project="usermass",
            parallel=True,
            region=visualise_region,
            rotation_center=unyt.unyt_array([x, y, z]),
            rotation_matrix=edge_on_rotation_matrix,
            boxsize=data.metadata.boxsize,
            backend="subsampled",
        )

        x_range = visualise_region[1] - visualise_region[0]
        y_range = visualise_region[3] - visualise_region[2]
        units = 1.0 / (x_range * y_range)
        # Unfortunately this is required to prevent us from {over,under}flowing
        # the units...
        units.convert_to_units(1.0 / (x_range.units * y_range.units))

        mass_map_edge = unyt.unyt_array(pixel_grid, units=units)
        mass_map_edge.convert_to_units(1.0 / unyt.pc ** 2)
        try:
            mass_map_edge[mass_map_edge == 0.0] = mass_map_edge[
                mass_map_edge > 0.0
            ].min()
        except:
            mass_map_edge[mass_map_edge == 0.0] = 1.0e-10

        try:
            H_kpc_gri[ilum] = calculate_scaleheight_fit(mass_map_edge.T, r_img_kpc)
        except:
            H_kpc_gri[ilum] = -1.0
        rgb_image_edge[:, :, ilum] = mass_map_edge.T

    print("H (gri): ", H_kpc_gri)
    image_edge = make_lupton_rgb(
        rgb_image_edge[:, :, 0],
        rgb_image_edge[:, :, 1],
        rgb_image_edge[:, :, 2],
        Q=10,
        stretch=0.5,
    )
    # mask with circle
    lx, ly = mass_map_edge.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    mask_circle = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    image_edge[mask_circle, :] = 255

    return image_face, image_edge, visualise_region, x, y, -1.0, H_kpc_gri


def get_stars_surface_density_map(
    catalogue,
    halo_id,
    plottype,
    snapshot_filename,
    size,
    npixloc,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
):
    # center of the halo
    x = catalogue.positions.xcmbp[halo_id]
    y = catalogue.positions.ycmbp[halo_id]
    z = catalogue.positions.zcmbp[halo_id]

    # angular momentum of the stars (for projection)
    lx = catalogue.angular_momentum.lx_star[halo_id]
    ly = catalogue.angular_momentum.ly_star[halo_id]
    lz = catalogue.angular_momentum.lz_star[halo_id]

    angular_momentum_vector = np.array([lx.value, ly.value, lz.value])
    angular_momentum_vector /= np.linalg.norm(angular_momentum_vector)

    # needs to be in comoving coordinates for the mask
    region = [
        [
            x / catalogue.scale_factor - size / catalogue.scale_factor,
            x / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
        [
            y / catalogue.scale_factor - size / catalogue.scale_factor,
            y / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
        [
            z / catalogue.scale_factor - size / catalogue.scale_factor,
            z / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
    ]

    visualise_region = [x - 0.5 * size, x + 0.5 * size, y - 0.5 * size, y + 0.5 * size]

    data_mask = mask(snapshot_filename)
    data_mask.constrain_spatial(region)
    data = load(snapshot_filename, mask=data_mask)
    data.stars.coordinates = data.stars.coordinates.to_physical()

    data.stars.smoothing_lengths = generate_smoothing_lengths(
        coordinates=data.stars.coordinates,
        boxsize=data.metadata.boxsize,
        kernel_gamma=kernel_gamma,
        neighbours=11,
        speedup_fac=1,
        dimension=3,
    )

    # Face on projection
    pixel_grid = project_pixel_grid(
        data.stars,
        resolution=int(npixloc),
        project="masses",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array([x, y, z]),
        rotation_matrix=face_on_rotation_matrix,
        boxsize=data.metadata.boxsize,
        backend="subsampled",
    )

    x_range = visualise_region[1] - visualise_region[0]
    y_range = visualise_region[3] - visualise_region[2]
    units = 1.0 / (x_range * y_range)
    # Unfortunately this is required to prevent us from {over,under}flowing
    # the units...
    units.convert_to_units(1.0 / (x_range.units * y_range.units))
    units *= getattr(data.stars, "masses").units

    mass_map_face = unyt.unyt_array(pixel_grid, units=units)
    mass_map_face.convert_to_units(unyt.Msun / unyt.pc ** 2)
    mass_map_face[mass_map_face == 0.0] = 1.0e-20

    pixelsize = (visualise_region[1] - visualise_region[0]) / float(npixloc)
    totalmass = np.sum(mass_map_face) * pixelsize * pixelsize
    totalmass.convert_to_units("Msun")

    # Edge on projection
    pixel_grid = project_pixel_grid(
        data.stars,
        resolution=int(npixloc),
        project="masses",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array([x, y, z]),
        rotation_matrix=edge_on_rotation_matrix,
        boxsize=data.metadata.boxsize,
        backend="subsampled",
    )

    x_range = visualise_region[1] - visualise_region[0]
    y_range = visualise_region[3] - visualise_region[2]
    units = 1.0 / (x_range * y_range)
    # Unfortunately this is required to prevent us from {over,under}flowing
    # the units...
    units.convert_to_units(1.0 / (x_range.units * y_range.units))
    units *= getattr(data.stars, "masses").units

    mass_map_edge = unyt.unyt_array(pixel_grid, units=units)
    mass_map_edge.convert_to_units(unyt.Msun / unyt.pc ** 2)
    try:
        mass_map_edge[mass_map_edge == 0.0] = mass_map_edge[mass_map_edge > 0.0].min()
    except:
        mass_map_edge[mass_map_edge == 0.0] = 1.0e-10

    # mask with circle
    lx, ly = mass_map_edge.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    mask_circle = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    mass_map_edge[mask_circle] = np.nan

    # mask with circle
    lx, ly = mass_map_face.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    mask_circle = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    mass_map_face[mask_circle] = np.nan

    return mass_map_face.T, mass_map_edge.T, visualise_region, x, y, totalmass


def get_gas_surface_density_map(
    catalogue,
    halo_id,
    plottype,
    snapshot_filename,
    size,
    npixlocal,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
):
    # center of the halo in physical coordinates
    x = catalogue.positions.xcmbp[halo_id]
    y = catalogue.positions.ycmbp[halo_id]
    z = catalogue.positions.zcmbp[halo_id]

    # angular momentum of the stars (for projection)
    lx = catalogue.angular_momentum.lx_star[halo_id]
    ly = catalogue.angular_momentum.ly_star[halo_id]
    lz = catalogue.angular_momentum.lz_star[halo_id]

    angular_momentum_vector = np.array([lx.value, ly.value, lz.value])
    angular_momentum_vector /= np.linalg.norm(angular_momentum_vector)

    # face_on_rotation_matrix = rotation_matrix_from_vector(
    #   angular_momentum_vector
    # )
    # edge_on_rotation_matrix = rotation_matrix_from_vector(
    #   angular_momentum_vector,
    #   axis="y"
    # )

    # needs to be in comoving coordinates for the mask
    region = [
        [
            x / catalogue.scale_factor - size / catalogue.scale_factor,
            x / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
        [
            y / catalogue.scale_factor - size / catalogue.scale_factor,
            y / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
        [
            z / catalogue.scale_factor - size / catalogue.scale_factor,
            z / catalogue.scale_factor + size / catalogue.scale_factor,
        ],
    ]

    visualise_region = [x - 0.5 * size, x + 0.5 * size, y - 0.5 * size, y + 0.5 * size]

    data_mask = mask(snapshot_filename)
    data_mask.constrain_spatial(region)
    data = load(snapshot_filename, mask=data_mask)
    data.gas.coordinates = data.gas.coordinates.to_physical()
    data.gas.smoothing_lengths.convert_to_physical()

    if plottype == "HI":
        data.gas.usermass = (
            data.gas.masses
            * data.gas.species_fractions.HI
            * data.gas.element_mass_fractions.hydrogen
        )
    elif plottype == "H2":
        data.gas.usermass = (
            data.gas.masses
            * 2.0
            * data.gas.species_fractions.H2
            * data.gas.element_mass_fractions.hydrogen
        )
    elif "GK11" in plottype:
        data.gas.usermass = (
            get_GK11fractions(data, plottype)
            * data.gas.masses
            * data.gas.element_mass_fractions.hydrogen
        )
    elif "hydrogen" in plottype:
        data.gas.usermass = data.gas.masses * data.gas.element_mass_fractions.hydrogen
    elif "helium" in plottype:
        data.gas.usermass = data.gas.masses * data.gas.element_mass_fractions.helium
    elif "totaloxygen" in plottype:
        data.gas.usermass = data.gas.masses * data.gas.element_mass_fractions.oxygen
    elif "diffuseoxygen" in plottype:
        data.gas.usermass = data.gas.diffuse_oxygen_masses_from_table
    elif "sfr" in plottype:
        data.gas.star_formation_rates.convert_to_units(unyt.Msun / unyt.yr)
        data.gas.star_formation_rates[data.gas.star_formation_rates < 0.0] = 0.0
        data.gas.usermass = data.gas.star_formation_rates
    else:
        print("Unknown plottype: ", plottype)
        import sys

        sys.exit()

    if not "sfr" in plottype:
        data.gas.usermass.convert_to_units("Msun")

    # Face on projection
    mass_map_face = project_gas(
        data,
        resolution=int(npixlocal),
        project="usermass",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array([x, y, z]),
        rotation_matrix=face_on_rotation_matrix,
        backend="subsampled",
    )

    if "sfr" in plottype:
        mass_map_face.convert_to_units(unyt.Msun / unyt.yr / unyt.pc ** 2)
    else:
        mass_map_face.convert_to_units(unyt.Msun / unyt.pc ** 2)

    pixelsize = (visualise_region[1] - visualise_region[0]) / float(npixlocal)
    totalmass = np.sum(mass_map_face) * pixelsize * pixelsize
    if "sfr" in plottype:
        totalmass.convert_to_units("Msun/yr")
    else:
        totalmass.convert_to_units("Msun")

    # Edge on projection
    mass_map_edge = project_gas(
        data,
        resolution=int(npixlocal),
        project="usermass",
        parallel=True,
        region=visualise_region,
        rotation_center=unyt.unyt_array([x, y, z]),
        rotation_matrix=edge_on_rotation_matrix,
        backend="subsampled",
    )

    if "sfr" in plottype:
        mass_map_edge.convert_to_units(unyt.Msun / unyt.yr / unyt.pc ** 2)
    else:
        mass_map_edge.convert_to_units(unyt.Msun / unyt.pc ** 2)

    # mask with circle
    lx, ly = mass_map_edge.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    mask_circle = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    mass_map_edge[mask_circle] = np.nan

    # mask with circle
    lx, ly = mass_map_face.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    mask_circle = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    mass_map_face[mask_circle] = np.nan

    return mass_map_face.T, mass_map_edge.T, visualise_region, x, y, totalmass


def get_radial_profile(mass_map, radialbin_kpc, pixsize_kpc, r_img_kpc):
    nbins_radial = int(r_img_kpc.value / radialbin_kpc)

    y, x = np.indices((mass_map.shape))
    npix = len(mass_map[0, :])
    pcenter = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    r = np.hypot(x - pcenter[0], y - pcenter[1])
    r_1D = r.ravel() * pixsize_kpc

    # from mass per pc-2 to mass per pixel
    mass_map_per_pixel = mass_map * pixsize_kpc * pixsize_kpc * unyt.kpc ** 2
    mass_map_per_pixel.convert_to_units(unyt.Msun)
    mass_map_1D = (mass_map_per_pixel.ravel()).value

    rmin = 0.0
    rmax = nbins_radial * radialbin_kpc

    M_hist, r_hist = np.histogram(
        r_1D, weights=mass_map_1D, bins=nbins_radial, range=(rmin, rmax), density=False
    )

    # calculate the area of each radial bin
    A = np.zeros(len(r_hist) - 1)
    for i in range(len(r_hist) - 1):
        A[i] = np.pi * (r_hist[i + 1] * r_hist[i + 1] - r_hist[i] * r_hist[i])

    A_unyt = A * unyt.kpc ** 2
    Sigma = M_hist * unyt.Msun / A_unyt
    Sigma.convert_to_units(unyt.Msun / unyt.pc ** 2)

    r_central = np.zeros(len(r_hist) - 1)
    for i in range(len(r_hist) - 1):
        r_central[i] = 0.5 * (r_hist[i + 1] + r_hist[i])

    r_central = r_central * unyt.kpc

    return r_central, Sigma


def plot_galaxy(
    catalogue,
    halo_id,
    index,
    snapshot_filename,
    face_on_rotation_matrix,
    edge_on_rotation_matrix,
    output_path,
):
    print(matplotlib.get_cachedir())

    fig = plt.figure(figsize=(15.0, 3.5))
    fig.subplots_adjust(left=0.01, right=0.95, top=0.85, bottom=0.12)
    gs = gridspec.GridSpec(
        2, nr_total_plots, wspace=0.0, hspace=0.15, height_ratios=[0.05, 1.0]
    )

    # General information
    text = "z = %.1f" % (catalogue.redshift) + "\n\n"
    text += r"${\bf " + "VR\ halo\ id:\ \ \ %3.3i" % (halo_id) + r"}$" + "\n"
    text += (
        r"M$_{\mathrm{200,crit}}$ = "
        + sci_notation(catalogue.masses.mass_200crit[halo_id].to("Msun").value)
        + r" M$_{\odot}$"
        + "\n"
    )
    text += (
        r"M$_{\mathrm{*,30kpc}}$ = "
        + sci_notation(catalogue.masses.mass_star_30kpc[halo_id].to("Msun").value)
        + r" M$_{\odot}$"
        + "\n"
    )
    text += (
        r"M$_{\mathrm{gas,30kpc}}$ = "
        + sci_notation(catalogue.masses.mass_gas_30kpc[halo_id].to("Msun").value)
        + r" M$_{\odot}$"
        + "\n"
    )
    text += (
        r"M$_{\mathrm{HI,30kpc}}$ = "
        + sci_notation(
            catalogue.gas_hydrogen_species_masses.HI_mass_30_kpc[halo_id]
            .to("Msun")
            .value
        )
        + r" M$_{\odot}$"
        + "\n"
    )
    text += (
        r"M$_{\mathrm{H2,30kpc}}$ = "
        + sci_notation(
            catalogue.gas_hydrogen_species_masses.H2_mass_30_kpc[halo_id]
            .to("Msun")
            .value
        )
        + r" M$_{\odot}$"
        + "\n"
    )
    sSFR = (
        catalogue.apertures.sfr_gas_100_kpc[halo_id]
        / catalogue.apertures.mass_star_100_kpc[halo_id]
    )
    print(halo_id, sSFR)
    if np.isfinite(sSFR):
        text += (
            r"sSFR$_{\mathrm{100}}$ = "
            + sci_notation(sSFR.to("Gyr**(-1)").value)
            + r" Gyr$^{-1}$"
        )

    ax = plt.subplot(gs[nr_total_plots])
    ax.set_aspect("equal")
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.05, 0.95, text, ha="left", va="top", transform=ax.transAxes, fontsize=8)
    ax.text(
        0.05,
        1.20,
        "%i Galaxy" % (index + 1),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )

    # Stars gri face-on
    ax = plt.subplot(gs[nr_total_plots + 1])
    ax.set_title("Stars (gri) - face")
    (
        mass_map_face,
        mass_map_edge,
        visualise_region,
        x,
        y,
        totalmass,
        H_kpc_gri,
    ) = get_stars_surface_brightness_map(
        catalogue,
        halo_id,
        snapshot_filename,
        size,
        npix,
        r_img_kpc,
        face_on_rotation_matrix,
        edge_on_rotation_matrix,
    )

    mass_map_face_plot = mass_map_face
    mass_map_edge_plot = mass_map_edge
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    im = ax.imshow(mass_map_face_plot, extent=visualise_region)
    circle = plt.Circle(
        (x, y),
        (0.99 * r_img_kpc.value) / 1000.0,
        color="black",
        fill=False,
        linewidth=2,
    )
    ax.add_artist(circle)
    ax.plot(
        [x - lbar_kpc / 2.0, x + lbar_kpc / 2.0],
        [y + ypos_bar, y + ypos_bar],
        color="white",
        linewidth=2,
        linestyle="solid",
    )
    ax.text(
        x,
        y + ypos_bar,
        "%i kpc" % (int(lbar_kpc.value)),
        color="white",
        verticalalignment="bottom",
        horizontalalignment="center",
    )
    # Stars gri edge-on
    ax = plt.subplot(gs[nr_total_plots + 2])
    ax.set_title("Stars (gri) - edge")
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    im = ax.imshow(mass_map_edge_plot, extent=visualise_region)
    circle = plt.Circle(
        (x, y),
        (0.99 * r_img_kpc.value) / 1000.0,
        color="black",
        fill=False,
        linewidth=2,
    )
    ax.add_artist(circle)
    ax.text(
        0.5,
        0.2,
        r"H$_{r}$ = %.2f kpc" % (H_kpc_gri[1]),
        ha="center",
        va="top",
        color="white",
        transform=ax.transAxes,
        fontsize=8,
    )
    # HI face-on
    ax = plt.subplot(gs[nr_total_plots + 3])
    ax.set_title("Gas (HI) - face")
    (
        mass_map_face,
        mass_map_edge,
        visualise_region,
        x,
        y,
        totalmass,
    ) = get_gas_surface_density_map(
        catalogue,
        halo_id,
        "HI",
        snapshot_filename,
        size,
        npix,
        face_on_rotation_matrix,
        edge_on_rotation_matrix,
    )
    mass_map_face.convert_to_units("Msun / pc**2")
    mass_map_edge.convert_to_units("Msun / pc**2")
    mass_map_face_plot_HI = mass_map_face  # save for H2 ratio plot
    totalmass_H2 = totalmass  # save for H2 ratio plot
    mass_map_face_plot = np.log10(mass_map_face.value)
    mass_map_edge_plot = np.log10(mass_map_edge.value)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    colormap = "Blues"
    cmap_loc = plt.get_cmap(colormap)
    ccolor = cmap_loc(0.5)
    im = ax.imshow(
        mass_map_face_plot, cmap=colormap, extent=visualise_region, vmin=vmin, vmax=vmax
    )
    circle = plt.Circle(
        (x, y), (0.99 * r_img_kpc.value) / 1000.0, color=ccolor, fill=False, linewidth=2
    )
    ax.add_artist(circle)
    # HI edge-on
    ax = plt.subplot(gs[nr_total_plots + 4])
    ax.set_title("Gas (HI) - edge")
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    im = ax.imshow(
        mass_map_edge_plot, cmap=colormap, extent=visualise_region, vmin=vmin, vmax=vmax
    )
    circle = plt.Circle(
        (x, y), (0.99 * r_img_kpc.value) / 1000.0, color=ccolor, fill=False, linewidth=2
    )
    ax.add_artist(circle)
    try:
        H_kpc = calculate_scaleheight_fit(mass_map_edge.value)
        ax.text(
            0.5,
            0.2,
            r"H$_{\mathrm{HI}}$ = %.2f kpc" % (H_kpc),
            ha="center",
            va="top",
            color="black",
            transform=ax.transAxes,
            fontsize=8,
        )
    except:
        H_kpc = -1.0

    # HI colorbar
    cax = plt.subplot(gs[3:5])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("log $\Sigma_{\mathrm{%s}}$ [M$_{\odot}$ pc$^{-2}$]" % ("HI"))
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_ticks(np.arange(round(vmin), vmax + 0.5, 0.5))
    # H2 face-on
    ax = plt.subplot(gs[nr_total_plots + 5])
    ax.set_title("Gas (H2) - face")
    (
        mass_map_face,
        mass_map_edge,
        visualise_region,
        x,
        y,
        totalmass,
    ) = get_gas_surface_density_map(
        catalogue,
        halo_id,
        "H2",
        snapshot_filename,
        size,
        npix,
        face_on_rotation_matrix,
        edge_on_rotation_matrix,
    )
    mass_map_face.convert_to_units("Msun / pc**2")
    mass_map_edge.convert_to_units("Msun / pc**2")
    mass_map_face_plot_H2 = mass_map_face  # save for H2 ratio plot
    mass_map_face_plot = np.log10(mass_map_face.value)
    mass_map_edge_plot = np.log10(mass_map_edge.value)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    colormap = "Greens"
    cmap_loc = plt.get_cmap(colormap)
    ccolor = cmap_loc(0.5)
    im = ax.imshow(
        mass_map_face_plot, cmap=colormap, extent=visualise_region, vmin=vmin, vmax=vmax
    )
    circle = plt.Circle(
        (x, y), (0.99 * r_img_kpc.value) / 1000.0, color=ccolor, fill=False, linewidth=2
    )
    ax.add_artist(circle)
    # H2 edge-on
    ax = plt.subplot(gs[nr_total_plots + 6])
    ax.set_title("Gas (H2) - edge")
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    im = ax.imshow(
        mass_map_edge_plot, cmap=colormap, extent=visualise_region, vmin=vmin, vmax=vmax
    )
    circle = plt.Circle(
        (x, y), (0.99 * r_img_kpc.value) / 1000.0, color=ccolor, fill=False, linewidth=2
    )
    ax.add_artist(circle)
    try:
        H_kpc = calculate_scaleheight_fit(mass_map_edge.value)
        ax.text(
            0.5,
            0.2,
            r"H$_{\mathrm{H2}}$ = %.2f kpc" % (H_kpc),
            ha="center",
            va="top",
            color="black",
            transform=ax.transAxes,
            fontsize=8,
        )
    except:
        H_kpc = -1.0

    # H2 colorbar
    cax = plt.subplot(gs[5:7])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("log $\Sigma_{\mathrm{%s}}$ [M$_{\odot}$ pc$^{-2}$]" % ("H2"))
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_ticks(np.arange(round(vmin), vmax + 0.5, 0.5))
    # Sigma H2 / Sigma HI vs. Sigma HI+H2
    ax = plt.subplot(gs[nr_total_plots + 7])
    ax.set_title("HI - H2 transition")
    ax.set_aspect(
        (Sigma_max - Sigma_min) / (np.log10(H2_over_HI_max) - np.log10(H2_over_HI_min))
    )
    ax.set_xlim(Sigma_min, Sigma_max)
    ax.set_ylim(H2_over_HI_min, H2_over_HI_max)
    ax.set_yscale("log")
    ax.set_yticks([0.1, 1.0, 10.0, 100.0])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlabel(
        r"$\Sigma_{\mathrm{HI}}$ + $\Sigma_{\mathrm{H2}}$ [M$_{\odot}$ pc$^{-2}$]"
    )
    ax.set_ylabel(r"$\Sigma_{\mathrm{H2}}$ / $\Sigma_{\mathrm{HI}}$")
    # Schruba+2011 data
    if totalmass_H2.to("Msun").value > 1.0e7:
        markers = ["o", "s", "D"]
        for irad in range(len(radialbinsizes)):
            radialbin_kpc = radialbinsizes[irad]
            # datapoint position
            _, Sigma_HIH2_sub = get_radial_profile(
                mass_map_face_plot_HI + mass_map_face_plot_H2,
                radialbin_kpc,
                pixsize_kpc,
                r_img_kpc,
            )
            _, Sigma_HI_sub = get_radial_profile(
                mass_map_face_plot_HI, radialbin_kpc, pixsize_kpc, r_img_kpc
            )
            _, Sigma_H2_sub = get_radial_profile(
                mass_map_face_plot_H2, radialbin_kpc, pixsize_kpc, r_img_kpc
            )
            # datapoint color (metallicity)
            (
                mass_map_face_plot_H,
                _,
                _,
                _,
                _,
                totalmass_H,
            ) = get_gas_surface_density_map(
                catalogue,
                halo_id,
                "hydrogen",
                snapshot_filename,
                size,
                npix,
                face_on_rotation_matrix,
                edge_on_rotation_matrix,
            )
            (
                mass_map_face_plot_O_diffuse,
                _,
                _,
                _,
                _,
                totalmass_O_diffuse,
            ) = get_gas_surface_density_map(
                catalogue,
                halo_id,
                "diffuseoxygen",
                snapshot_filename,
                size,
                npix,
                face_on_rotation_matrix,
                edge_on_rotation_matrix,
            )
            (
                mass_map_face_plot_star,
                _,
                _,
                _,
                _,
                totalmass_star,
            ) = get_stars_surface_density_map(
                catalogue,
                halo_id,
                "stars",
                snapshot_filename,
                size,
                npix,
                face_on_rotation_matrix,
                edge_on_rotation_matrix,
            )

            _, Sigma_H_sub = get_radial_profile(
                mass_map_face_plot_H, radialbin_kpc, pixsize_kpc, r_img_kpc
            )
            _, Sigma_O_diffuse_sub = get_radial_profile(
                mass_map_face_plot_O_diffuse, radialbin_kpc, pixsize_kpc, r_img_kpc
            )
            _, Sigma_star = get_radial_profile(
                mass_map_face_plot_star, radialbin_kpc, pixsize_kpc, r_img_kpc
            )
            sc = ax.scatter(
                Sigma_HIH2_sub[Sigma_HI_sub > 0.0],
                Sigma_H2_sub[Sigma_HI_sub > 0.0] / Sigma_HI_sub[Sigma_HI_sub > 0.0],
                c=12.0
                + np.log10(
                    Sigma_O_diffuse_sub[Sigma_HI_sub > 0.0]
                    / Sigma_H_sub[Sigma_HI_sub > 0.0]
                    / 16.0
                ),
                vmin=twelve_plus_logOH_min,
                vmax=twelve_plus_logOH_max,
                cmap=cmRdBu,
                edgecolors="black",
                label="Azim.avg. %.2f kpc" % (radialbin_kpc),
                marker=markers[irad],
            )
    ax.legend(bbox_to_anchor=(-0.1, -0.1), ncol=3, loc="upper right", fontsize=8)

    # Metallicity colorbar
    if totalmass_H2.to("Msun").value > 1.0e7:
        cax = plt.subplot(gs[7])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label(r"12 + log$_{\mathrm{10}}$(O/H)$_{\mathrm{diffuse}}$")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.set_ticks(
            [twelve_plus_logOH_min, twelve_plus_logOH_solar, twelve_plus_logOH_max]
        )

    Zdiffuse = catalogue.cold_dense_gas_properties.cold_dense_diffuse_metal_mass_100_kpc[
        halo_id
    ]
    ColdGas = catalogue.cold_dense_gas_properties.cold_dense_gas_mass_100_kpc[halo_id]
    twelve_plus_logOH = -99.0
    if ColdGas > 0.0:
        twelve_plus_logOH = (
            np.log10(Zdiffuse / (Zsun * ColdGas)) + twelve_plus_log_OH_solar
        )

    ax.text(
        0.95,
        0.05,
        "12 + log$_{\mathrm{10}}$(O/H) = %.2f" % (twelve_plus_logOH),
        va="bottom",
        ha="right",
        transform=ax.transAxes,
        fontsize=8,
    )

    outputname = f"{output_path}/surface_overview_halo{index:03d}.png"
    print(outputname)
    fig.savefig(outputname, dpi=150)
    plt.close()

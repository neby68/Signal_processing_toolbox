import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, List

def topoplot_indie(Values: np.ndarray, chanlocs: List[dict], headrad: float = 0.5, 
                   grid_scale: int = 67, circgrid: int = 201, headcolor: List[int] = [0, 0, 0], 
                   hlinewidth: float = 1.7, blankingringwidth: float = 0.035, headringwidth: float = 0.007, 
                   plotrad: float = 0.6, shading: str = 'interp', contournum: int = 6, electrodes: str = 'on') -> Tuple[plt.Axes, List[int], np.ndarray]:
    """
    Plot topographic map of independent values on scalp.

    Parameters:
    - Values (numpy.ndarray): Independent values for each electrode.
    - chanlocs (List[dict]): List of dictionaries containing channel locations.
    - headrad (float): Actual head radius.
    - grid_scale (int): Number of points in the grid for plotting.
    - circgrid (int): Number of angles to use in drawing circles.
    - headcolor (List[int]): RGB values for the default head color.
    - hlinewidth (float): Default linewidth for head, nose, and ears.
    - blankingringwidth (float): Width of the blanking ring.
    - headringwidth (float): Width of the cartoon head ring.
    - plotrad (float): Radius of the plotting circle.
    - shading (str): Shading type ('flat' or 'interp').
    - contournum (int): Number of contours in the plot.
    - electrodes (str): Display mode for electrodes ('on', 'labels', 'numbers').

    Returns:
    - handle (plt.Axes): Matplotlib Axes object.
    - pltchans (List[int]): Indices of channels plotted.
    - epos (numpy.ndarray): Electrode positions.
    """

    # Set defaults
    GRID_SCALE = grid_scale
    CIRCGRID = circgrid
    HEADCOLOR = headcolor
    HLINEWIDTH = hlinewidth
    BLANKINGRINGWIDTH = blankingringwidth
    HEADRINGWIDTH = headringwidth

    plotrad = plotrad
    Values = Values.astype(float)

    if shading not in ['flat', 'interp']:
        raise ValueError('Invalid shading parameter')

    Values = Values.flatten()  # make Values a column vector

    # Read channel location
    labels = [chan['labels'] for chan in chanlocs]
    Th = [chan['theta'] for chan in chanlocs]
    Rd = [chan['radius'] for chan in chanlocs]

    Th = np.deg2rad(Th)  # convert degrees to radians
    allchansind = list(range(len(Th)))
    plotchans = list(range(len(chanlocs)))

    # Remove infinite and NaN values
    inds = np.union1d(np.where(np.isnan(Values))[0], np.where(np.isinf(Values))[0])  # NaN and Inf values
    for chani, chanloc in enumerate(chanlocs):
        if not chanloc['X']:
            inds = np.append(inds, chani)

    plotchans = np.setdiff1d(plotchans, inds)

    x, y = np.pol2cart(Th, Rd)  # transform electrode locations from polar to cartesian coordinates
    plotchans = np.abs(plotchans)  # reverse indicated channel polarities
    allchansind = np.array(allchansind)[plotchans]
    Th = np.array(Th)[plotchans]
    Rd = np.array(Rd)[plotchans]
    x = np.array(x)[plotchans]
    y = np.array(y)[plotchans]
    labels = np.array(labels)[plotchans]  # remove labels for electrodes without locations
    Values = Values[plotchans]
    intrad = min(1.0, max(Rd) * 1.02)  # default: just outside the outermost electrode location

    # Find plotting channels
    pltchans = np.where(Rd <= plotrad)[0]  # plot channels inside plotting circle
    intchans = np.where((x <= intrad) & (y <= intrad))[0]  # interpolate and plot channels inside interpolation square

    # Eliminate channels not plotted
    allx, ally = x, y
    allchansind = allchansind[pltchans]
    intTh = Th[intchans]  # eliminate channels outside the interpolation area
    intRd = Rd[intchans]
    intx = x[intchans]
    inty = y[intchans]
    Th = Th[pltchans]  # eliminate channels outside the plotting area
    Rd = Rd[pltchans]
    x = x[pltchans]
    y = y[pltchans]

    intValues = Values[intchans]
    Values = Values[pltchans]

    labels = labels.astype(str)

    # Squeeze channel locations to <= headrad
    squeezefac = headrad / plotrad
    intRd = intRd * squeezefac  # squeeze electrode arc_lengths towards the vertex
    Rd = Rd * squeezefac  # squeeze electrode arc_lengths towards the vertex
    intx = intx * squeezefac
    inty = inty * squeezefac
    x = x * squeezefac
    y = y * squeezefac
    allx = allx * squeezefac
    ally = ally * squeezefac

    # Create grid
    xmin, xmax = min(-headrad, min(intx)), max(headrad, max(intx))
    ymin, ymax = min(-headrad, min(inty)), max(headrad, max(inty))
    xi = np.linspace(xmin, xmax, GRID_SCALE)  # x-axis description (row vector)
    yi = np.linspace(ymin, ymax, GRID_SCALE)  # y-axis description (row vector)

    Xi, Yi, Zi = np.meshgrid(xi, yi, 0)
    interp_values = np.interp(xi, inty, intValues)  # interpolate data
    for i in range(GRID_SCALE):
        Zi[i, :, :] = interp_values

    # Mask out data outside the head
    mask = np.sqrt(Xi**2 + Yi**2) <= headrad  # mask outside the plotting circle
    Zi[~mask] = np.nan  # mask non-plotting voxels with NaNs
    grid = plotrad  # unless 'noplot', then 3rd output arg is plotrad
    delta = xi[1] - xi[0]  # length of grid entry

    # Scale the axes and make the plot
    plt.clf()  # clear current axis
    ax = plt.gca()  # uses current axes
    AXHEADFAC = 1.05  # do not leave room for external ears if head cartoon
    ax.set_xlim([-headrad * AXHEADFAC, headrad * AXHEADFAC])
    ax.set_ylim([-headrad * AXHEADFAC, headrad * AXHEADFAC])
    unsh = (GRID_SCALE + 1) / GRID_SCALE  # un-shrink the effects of 'interp' SHADING

    if shading == 'interp':
        handle = ax.pcolormesh(Xi * unsh, Yi * unsh, Zi[0, :, :], shading=shading, edgecolor='none', cmap='viridis')
    else:
        handle = ax.pcolormesh(Xi - delta / 2, Yi - delta / 2, Zi[0, :, :], shading=shading, edgecolor='none', cmap='viridis')

    ax.contour(Xi, Yi, Zi[0, :, :], CONTOURNUM, colors='k', linewidths=0.5, hittest=False)

    # Plot filled ring to mask jagged grid boundary
    hwidth = HEADRINGWIDTH  # width of head ring
    hin = squeezefac * headrad * (1 - hwidth / 2)  # inner head ring radius

    if shading == 'interp':
        rwidth = BLANKINGRINGWIDTH * 1.3  # width of blanking outer ring
    else:
        rwidth = BLANKINGRINGWIDTH  # width of blanking outer ring

    rin = headrad * (1 - rwidth / 2)  # inner ring radius
    if hin > rin:
        rin = hin  # don't blank inside the head ring

    circ = np.linspace(0, 2 * np.pi, CIRCGRID)
    rx, ry = np.sin(circ), np.cos(circ)
    ringx = np.concatenate([rx * (rin + rwidth), rx * rin])
    ringy = np.concatenate([ry * (rin + rwidth), ry * rin])
    ringh = ax.fill(ringx, ringy, color='w', edgecolor='none', zorder=0.01)  # plot the filled ring to mask jagged grid boundary

    # Plot cartoon head, ears, nose
    headx = np.concatenate([rx * (hin + hwidth), rx * hin])
    heady = np.concatenate([ry * (hin + hwidth), ry * hin])
    ax.fill(headx, heady, color=HEADCOLOR, edgecolor=HEADCOLOR, zorder=0.01)  # plot the filled cartoon head

    # Plot ears and nose
    base = headrad - 0.0046
    basex = 0.18 * headrad  # nose width
    tip = 1.15 * headrad
    tiphw = 0.04 * headrad  # nose tip half width
    tipr = 0.01 * headrad  # nose tip rounding
    q = 0.04  # ear lengthening
    earx = [0.497 - 0.005, 0.510, 0.518, 0.5299, 0.5419, 0.54, 0.547, 0.532, 0.510, 0.489 - 0.005]  # headrad = 0.5
    eary = [q + 0.0555, q + 0.0775, q + 0.0783, q + 0.0746, q + 0.0555, -0.0055, -0.0932, -0.1313, -0.1384, -0.1199]
    sf = headrad / plotrad  # squeeze the model ears and nose by this factor
    earx = np.array(earx) * sf
    eary = np.array(eary) * sf

    ax.plot(earx, eary, 'k-', linewidth=HLINEWIDTH, zorder=0.01)  # plot left ear
    ax.plot(-earx, eary, 'k-', linewidth=HLINEWIDTH, zorder=0.01)  # plot right ear

    # Mark electrode locations
    if electrodes == 'on':  # plot electrodes as spots
        ax.plot(y, x, 'ko', markersize=5, linewidth=0.5, zorder=0.01)
    elif electrodes == 'labels':  # print electrode names (labels)
        for i, label in enumerate(labels):
            ax.text(float(y[i]), float(x[i]), label, ha='center', va='middle', color='k', zorder=0.01)
    elif electrodes == 'numbers':
        for i, chanind in enumerate(allchansind):
            ax.text(float(y[i]), float(x[i]), str(chanind), ha='center', va='middle', color='k', zorder=0.01)

    epos = np.array([x, y])
    ax.axis('off')
    ax.set_aspect('equal')

    return ax, allchansind, epos

# Example usage:
# topoplot_indie(Values, chanlocs)

if __name__ == "__main__":
        
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    # (Assuming the topoplot_indie function is available here)

    # Load sample EEG data
    mat_data = loadmat(r'src\time_series_noise_simulation\sampleEEGdata.mat')

    # Explore the data
    EEG = mat_data['EEG']

    # Plot ERPs
    erp = np.mean(EEG['data'], axis=2)

    # Pick a channel and plot ERP
    chan2plot = 'fcz'
    chan_idx = np.where(np.array(EEG['chanlocs']['labels']) == chan2plot)[0][0]

    plt.figure(figsize=(8, 6))
    plt.plot(EEG['times'][0], erp[chan_idx, :], linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.title(f'ERP for Channel {chan2plot}')
    plt.show()

    # Plot topographical maps
    time2plot = 300  # in ms

    # Convert time in ms to time in indices
    tidx = np.argmin(np.abs(EEG['times'][0] - time2plot))

    # Get the values for the topoplot from the ERP
    values_for_topoplot = erp[:, tidx]

    # Plot topographical map
    ax, pltchans, epos = topoplot_indie(values_for_topoplot, EEG['chanlocs'][0])

    # Additional customization for the topoplot
    ax.set_title(f'Topographical Map at {time2plot} ms')
    plt.show()


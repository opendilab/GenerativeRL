import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(data: np.ndarray, save_path: str, size=None, dpi=500):
    """
    Overview:
        Plot a grid of N x N subplots where:
        - Diagonal contains 1D histograms for each feature.
        - Off-diagonal contains 2D histograms (pcolormesh) showing relationships between pairs of features.
        - The colorbar of the 2D histograms shows percentages of total data points.

    Parameters:
    - data: numpy.ndarray of shape (B, N), where B is the number of samples and N is the number of features.
    - save_path: str, path to save the generated figure.
    - size: tuple (width, height), optional, size of the figure.
    - dpi: int, optional, resolution of the saved figure in dots per inch.
    """

    B, N = data.shape  # B: number of samples, N: number of features

    # Create a figure with N * N subplots
    fig, axes = plt.subplots(N, N, figsize=size if size else (4 * N, 4 * N))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # First, calculate the global minimum and maximum for the 2D histograms (normalized as percentages)
    hist_range = [
        [np.min(data[:, i]) * 1.02, np.max(data[:, i] * 1.02)] for i in range(N)
    ]
    global_min, global_max = float("inf"), float("-inf")

    # Loop to calculate the min and max percentage values across all 2D histograms
    for i in range(N):
        for j in range(N):
            if i != j:
                hist, xedges, yedges = np.histogram2d(
                    data[:, j],
                    data[:, i],
                    bins=30,
                    range=[hist_range[j], hist_range[i]],
                )
                hist_percentage = hist / B * 100  # Convert counts to percentages
                global_min = min(global_min, hist_percentage.min())
                global_max = max(global_max, hist_percentage.max())

    # Second loop to plot the figures using pcolormesh
    for i in range(N):
        for j in range(N):
            if i == j:
                # Diagonal: plot 1D histogram for feature i
                if N == 1:
                    axes.hist(data[:, i], bins=30, color="skyblue", edgecolor="black")
                else:
                    axes[i, j].hist(
                        data[:, i], bins=30, color="skyblue", edgecolor="black"
                    )
                # axes[i, j].set_title(f'Hist of Feature {i+1}')
            else:
                # Off-diagonal: calculate 2D histogram and plot using pcolormesh with unified color scale (as percentage)
                hist, xedges, yedges = np.histogram2d(
                    data[:, j],
                    data[:, i],
                    bins=30,
                    range=[hist_range[j], hist_range[i]],
                )
                hist_percentage = hist / B * 100  # Convert to percentage

                # Use pcolormesh to plot the 2D histogram
                mesh = axes[i, j].pcolormesh(
                    xedges,
                    yedges,
                    hist_percentage.T,
                    cmap="Blues",
                    vmin=global_min,
                    vmax=global_max,
                )
                axes[i, j].set_xlabel(f"Dimension {j+1}")
                axes[i, j].set_ylabel(f"Dimension {i+1}")

    if N > 1:
        # Add a single colorbar for all pcolormesh plots (showing percentage)
        cbar = fig.colorbar(
            mesh, ax=axes, orientation="vertical", fraction=0.02, pad=0.04
        )
        cbar.set_label("Percentage (%)")

    # Save the figure to the provided path
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_histogram2d_x_y(x_data, y_data, save_path: str, size=None, dpi=500):
    # Set up a figure with 3 subplots: 2D histogram, KDE, and scatter plot
    if isinstance(x_data, list):
        x_data = np.array(x_data)
    if isinstance(y_data, list):
        y_data = np.array(y_data)
    global_min, global_max = float("inf"), float("-inf")
    fig, ax = plt.subplots(figsize=size if size else (8, 6))
    x_max = ((x_data.max() + 99) // 100) * 100
    y_max = np.ceil(y_data.max() / 2) * 2
    y_min = (y_data.min() // 2) * 2
    # 2D Histogram for density
    hist2d, xedges, yedges = np.histogram2d(
        x_data, y_data, bins=100, range=[[0, x_max], [y_min, y_max]]
    )
    hist_percentage = hist2d / hist2d.sum()  # Normalize the histogram
    global_min = min(global_min, hist_percentage.min())
    global_max = max(global_max, hist_percentage.max())
    # Plot the 2D histogram
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        hist_percentage.T,
        cmap="Blues",
        vmin=global_min,
        vmax=global_max,
    )
    ax.set_xlabel("Returns")
    ax.set_ylabel("LogP")
    ax.set_title("2D Histogram Density Plot")

    # Add colorbar to the 2D histogram
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cb.set_label("Percentage (%)")

    # Save the plot
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)

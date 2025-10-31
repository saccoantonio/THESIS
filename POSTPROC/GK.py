import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
import scienceplots # type: ignore
from tqdm import tqdm # type: ignore
import gc
import glob
import os

plt.style.use(["science", "nature"])
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "figure.figsize": (8.0, 6.2),
        "axes.linewidth": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "grid.alpha": 0.25,
    }
)

# Simple methods

def autocorrelation_fft(x):
    """Compute the autocorrelation function using FFT."""
    
    # Remove the mean from the data to center it around zero
    x = x - np.mean(x)

    # Get the number of data points
    N = len(x)
    
    # Compute the Fast Fourier Transform (FFT) of x, 
    # padding to length 2N to reduce circular convolution (aliasing) effects
    fft_x = np.fft.fft(x, n=2*N)
    
    # Compute the inverse FFT of the power spectrum (fft_x * conjugate(fft_x)),
    # which gives the autocorrelation function (real part only, first N elements)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:N].real
    
    # Normalize the autocorrelation so that acf[0] = 1,
    # dividing by the variance and the decreasing number of overlapping points
    acf /= (x.var() * (N - np.arange(N)))
    
    # Return the normalized autocorrelation function
    return acf


def plot_acf(t, ac, acfit):
    """Plot the autocorrelation functions in 3 directions (or total vs fitted)."""
    
    # Create a new figure with a specific size (13x5 inches)
    plt.figure(figsize=(13, 5))
    
    # Plot the total autocorrelation function versus time
    plt.plot(t, ac, label="total acf")
    
    # Plot the fitted autocorrelation function versus time
    plt.plot(t, acfit, label="fit acf")
    
    # Draw a horizontal dashed line at y=0 for visual reference
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set the y-axis label using LaTeX syntax for the correlation formula
    plt.ylabel(r"$\frac{<J_i(0)\cdot J_i(t)>}{<J_i(0) \cdot J_i(0)>}$", fontsize=20)
    
    # Set the x-axis label to indicate correlation time in picoseconds
    plt.xlabel("correlation length (ps)", fontsize=12)
    
    # Use a logarithmic scale for the x-axis to better show decaying behavior
    plt.xscale('log')
    
    # Add a legend in the lower left corner to distinguish the curves
    plt.legend(loc="lower left")

    # Display the plot
    plt.show()


def ACF_fit(x, A, B, C, n, m):
    """Double exponential fit for the autocorrelation function (ACF)."""
    
    # Avoid division by zero or extremely small denominators for parameter n
    # If |n| is smaller than 1e-12, replace it with ±1e-12 preserving the sign
    if np.abs(n) < 1e-12: 
        n = np.sign(n) * 1e-12
    
    # Same safeguard for parameter m
    if np.abs(m) < 1e-12: 
        m = np.sign(m) * 1e-12
    
    # Compute the exponent arguments for the two exponential terms
    # np.clip limits values between -700 and 700 to avoid numerical overflow
    z1 = np.clip(-x/n, -700, 700)
    z2 = np.clip(-x/m, -700, 700)
    
    # Return the sum of two exponential decays plus a constant offset C
    # A and B are amplitudes, n and m are decay constants
    return A * np.exp(z1) + B * np.exp(z2) + C


def fit_acf(x, ac):
    """Fit the autocorrelation function (ACF) using a double exponential model."""
    
    # Perform a nonlinear least-squares fit of the ACF data (ac vs x)
    # using the ACF_fit function defined earlier as the model.
    # 'pars' contains the best-fit parameters [A, B, C, n, m]
    # 'cov' is the covariance matrix of the parameters (not used here).
    # maxfev=10000 increases the maximum number of function evaluations 
    # to help the optimizer converge for complex fits.
    pars, cov = curve_fit(ACF_fit, x, ac, maxfev=10000)
    
    # Return only the fitted parameters
    return pars


def compute_F_E(ac, J, x, delta=1000, n_plot=100000, prominence=0.0007):
    """
    Compute the fluctuation function F(t) and the averaged autocorrelation E(t) 
    from the autocorrelation function (ac) and time series J.
    
    Parameters:
        ac : array-like
            Autocorrelation function values.
        J : array-like
            Original time series data (e.g., flux).
        x : array-like or None
            Time array corresponding to ac and J.
        delta : int
            Window length for local averaging/variance.
        n_plot : int
            Maximum number of points to process for F and E.
        prominence : float
            Minimum prominence to detect peaks in F.
    
    Returns:
        F : array
            Fluctuation function F(t).
        E : array
            Local mean of the autocorrelation function E(t).
        first_peak_x : float or None
            x-coordinate of the first significant peak in F(t).
        first_zero_x : float or None
            x-coordinate of the first zero crossing of E(t).
    """
    
    # Initialize F array
    F = np.zeros(len(J))
    
    # Scale autocorrelation by variance of J
    cor = ac * J.var()
    
    # Compute fluctuation function F(t) using a moving window of length delta
    for j in range(min(len(J) - delta, n_plot)):
        # Local fluctuation: ratio of standard deviation to mean in the window
        F[j] = np.abs(np.sqrt(cor[j:j+delta].var()) / np.mean(cor[j:j+delta]))
    
    # Initialize E array
    E = np.zeros(len(ac))
    
    # Compute local mean of the autocorrelation function E(t)
    for j in range(min(len(ac) - delta, n_plot)):
        E[j] = np.mean(ac[j:j+delta])
    
    # --- Initialize output variables for first peak and zero crossing ---
    first_peak_x = None
    first_zero_x = None
    
    if x is not None:
        # ---- FIND FIRST PEAK OF F(t) ----
        # Normalize F for peak detection
        F_norm = F[:n_plot] / np.max(F[:n_plot])
        
        # Use scipy.signal.find_peaks to locate peaks above given prominence
        peaks, _ = find_peaks(F_norm, prominence=prominence)
        
        if len(peaks) > 0:
            # Take the first significant peak
            first_peak_idx = peaks[0]
            first_peak_x = x[first_peak_idx]
        else:
            print("No significant peak found in F(t).")
        
        # ---- FIND FIRST ZERO CROSSING OF E(t) ----
        # Detect sign changes (zero crossings) in E
        sign_change_idx = np.where(np.diff(np.sign(E[:n_plot])))[0]
        
        if len(sign_change_idx) > 0:
            # Take the first zero crossing and interpolate linearly
            first_zero_idx = sign_change_idx[0]
            x0, x1 = x[first_zero_idx], x[first_zero_idx+1]
            y0, y1 = E[first_zero_idx], E[first_zero_idx+1]
            first_zero_x = x0 - y0*(x1 - x0)/(y1 - y0)  # linear interpolation
    
    # Return fluctuation function, local mean, first peak, and first zero
    return F, E, first_peak_x, first_zero_x


def plot_F_E(x, F, first_peak_x, E, first_zero_x, n_plot=100000):
    """
    Plot the fluctuation function F(t) and the averaged autocorrelation E(t)
    with markers for the first significant peak and first zero crossing.
    
    Parameters:
        x : array-like
            Time or correlation length array.
        F : array-like
            Fluctuation function F(t).
        first_peak_x : float
            x-coordinate of the first significant peak in F(t).
        E : array-like
            Local mean of autocorrelation function E(t).
        first_zero_x : float
            x-coordinate of the first zero crossing of E(t).
        n_plot : int
            Number of points to plot.
    """
    
    # Create a figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    
    # --- Plot F(t) on the left subplot ---
    ax[0].plot(x[:n_plot], F[:n_plot])
    ax[0].set_ylabel("$F(t)$", fontsize=20)
    ax[0].set_xlabel("correlation length (ps)", fontsize=12)
    
    # Use logarithmic scale for both axes to show wide range
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    # Set y-axis limits to avoid extreme values
    ax[0].set_ylim([0.1, 100])
    
    # Mark the first significant peak with a red dot
    ax[0].plot(first_peak_x, F[0], "ro", label="Significant peak")
    
    # Add legend
    ax[0].legend()
    
    # Print the x-coordinate of the first peak
    print(f"Significant peak F(t): x = {first_peak_x:.4f} ps")
    
    # --- Plot E(t) on the right subplot ---
    ax[1].plot(x[:n_plot], E[:n_plot])
    
    # Draw horizontal line at zero for reference
    ax[1].axhline(y=0, color='black')
    
    ax[1].set_ylabel("$E(t)$", fontsize=20)
    ax[1].set_xlabel("correlation length (ps)", fontsize=12)
    
    # Limit x-axis to first 20 ps for better visualization
    ax[1].set_xlim(0, 20)
    
    # Mark the first zero crossing with a red dot
    ax[1].plot(first_zero_x, 0, "ro", label="First zero")
    
    # Add legend
    ax[1].legend()
    
    # Print the x-coordinate of the first zero crossing
    print(f"First zero E(t): x = {first_zero_x:.4f} ps")
    
    # Adjust subplot layout for better spacing
    plt.tight_layout()
    
    # Display the figure
    plt.show()


"""def integrate_acf(ac, dt=0.001):
    k = np.zeros(len(ac))
    for j in range(1, len(ac)):
        k[j] = dt/2*(ac[j]+ac[j-1])
    kk = np.zeros(len(ac))
    for j in range(1, len(ac)):
        kk[j] = np.sum(k[:j])
    return kk"""


def integrate_acf(ac, dt=0.001):
    """
    Numerically integrate the autocorrelation function ACF(t) using the trapezoidal rule.
    Vectorized version for high efficiency (O(N) instead of O(N²)).
    
    Parameters:
        ac : array-like
            Autocorrelation function values.
        dt : float
            Time step between consecutive ACF points.
    
    Returns:
        kk : array
            Cumulative integral of ACF(t) at each time point.
    """
    
    # Compute the area between consecutive points using the trapezoidal rule:
    # 0.5 * dt * (f_i + f_{i+1}) for each interval
    k = 0.5 * dt * (ac[1:] + ac[:-1])
    
    # Compute the cumulative sum to get the integral at each point
    # Prepend 0.0 to indicate the integral at the first point is zero
    kk = np.concatenate(([0.0], np.cumsum(k)))
    
    # Return the cumulative integral
    return kk


def compute_kappa(kk, J, V, T):
    """
    Compute the thermal conductivity κ from the integrated autocorrelation function.
    
    Parameters:
        kk : array-like
            Cumulative integral of the heat flux autocorrelation (from integrate_acf).
        J : array-like
            Original heat flux time series.
        V : float
            Volume of the system.
        T : float
            Temperature of the system (in Kelvin).
    
    Returns:
        kappa : array
            Thermal conductivity in W/mK (SI units).
    """
    
    # Scale factor according to the Green-Kubo formula:
    # kappa = (1 / (V * T^2)) * integral(<J(0) J(t)>) * conversion factor
    # 8.61673324e-5 is the Boltzmann constant in eV/K
    # Multiply by variance of J to scale the normalized autocorrelation
    # V is in the denominator beacuse J is not normalized by V
    scale = 1 / (V * T**2) / (8.61673324*10**(-5)) * J.var()
    
    # Conversion factor from eV/(ps·Å²·K) to W/mK
    metal2SI = 1.602176634*10**3
    
    # Return the thermal conductivity in SI units
    return kk * scale * metal2SI


def compute_kappa_FA_FD(x, kap, first_peak_x, first_zero_x, timestep):
    """
    Compute thermal conductivity using two approximate methods:
    FA (First Avalanche) and FD (First Dip).
    
    Parameters:
        x : array-like
            Time or correlation length array.
        kap : array-like
            Thermal conductivity as a function of time.
        first_peak_x : float or None
            Time of first significant peak in F(t).
        first_zero_x : float or None
            Time of first zero crossing in E(t).
        timestep : float
            Time step between consecutive points.
    
    Returns:
        tau_FA : float or None
            Time used for FA method.
        kappa_FA : float or None
            Thermal conductivity from FA method.
        tau_FD : float
            Time used for FD method.
        kappa_FD : float
            Thermal conductivity from FD method.
    """
    
    # --- FA (First Avalanche) Method ---
    tau_FA = None
    kappa_FA = None
    if first_peak_x is not None:
        # Set the time of the first peak as tau_FA
        tau_FA = first_peak_x
        # Get the corresponding thermal conductivity value from kap array
        kappa_FA = kap[int(first_peak_x / timestep)]
    
    # --- FD (First Dip) Method ---
    if first_zero_x is not None:
        # Set the time of first zero crossing as tau_FD
        tau_FD = first_zero_x
        # Get the corresponding thermal conductivity value
        kappa_FD = kap[int(first_zero_x / timestep)]
    else:
        # Fallback: if no zero crossing, take the maximum of the kappa(t) curve
        fd_idx = np.argmax(kap)
        tau_FD = x[fd_idx]
        kappa_FD = kap[fd_idx]
    
    # Return both FA and FD times and thermal conductivities
    return tau_FA, kappa_FA, tau_FD, kappa_FD


def plot_kappa_FA_FD(x, kap, fitkap, tau_FA, kappa_FA, tau_FD, kappa_FD, n_plot=100000):
    """
    Plot the thermal conductivity κ(t) from raw and fitted integration,
    marking FA (First Avalanche) and FD (First Dip) methods.
    
    Parameters:
        x : array-like
            Time or correlation length array.
        kap : array-like
            Thermal conductivity from raw integration.
        fitkap : array-like
            Thermal conductivity from double-exponential fit integration.
        tau_FA, kappa_FA : float
            Time and κ value for FA method.
        tau_FD, kappa_FD : float
            Time and κ value for FD method.
        n_plot : int
            Number of points to plot.
    """
    
    # Create a figure with specific size
    plt.figure(figsize=(13, 5))
    
    # Plot raw thermal conductivity (from cumulative integration)
    plt.plot(x[:n_plot], kap[:n_plot], label="raw integration")
    
    # Plot thermal conductivity obtained from double exponential fit
    plt.plot(x[:n_plot], fitkap[:n_plot], label="double exponential fit integration")
    
    # Mark FA method point with red circle and label
    plt.plot(tau_FA, kappa_FA, "ro", 
             label=f"FA: τc={tau_FA:.2f} ps, κ={kappa_FA:.3f} W/mK")
    
    # Mark FD method point with green square and label
    plt.plot(tau_FD, kappa_FD, "gs", 
             label=f"FD: τc={tau_FD:.2f} ps, κ={kappa_FD:.3f} W/mK")
    
    # Set y-axis label with LaTeX formatting
    plt.ylabel(r"$\kappa \; \frac{J}{m\cdot s \cdot K}$", fontsize=16)
    
    # Set x-axis label
    plt.xlabel("correlation length (ps)", fontsize=12)
    
    # Add legend to distinguish curves and points
    plt.legend()
    
    # Display the plot
    plt.show()
    
    # Print FA and FD method results in the console
    print(f"Metodo FA → τc = {tau_FA:.2f} ps, κ = {kappa_FA:.3f} W/mK")
    print(f"Metodo FD → τc = {tau_FD:.2f} ps, κ = {kappa_FD:.3f} W/mK")


def compute_kappa_Chen(pars, J, V, T, tau_FA):
    """
    Compute thermal conductivity using Chen's method with double-exponential fit parameters.
    
    Parameters:
        pars : list or array-like
            Fit parameters from double exponential fit: [A1, A2, Y0, tau1, tau2]
        J : array-like
            Heat flux time series.
        V : float
            Volume of the system.
        T : float
            Temperature (Kelvin).
        tau_FA : float
            Correlation time from FA (First Avalanche) method.
    
    Returns:
        kappa_C : float
            Thermal conductivity computed from Chen's method without Y0 contribution.
        kappa_F : float
            Thermal conductivity including Y0 * tau_FA contribution.
    """
    
    # Unpack double-exponential fit parameters
    A1, A2, Y0, tau1, tau2 = pars
    
    # Scale factor according to Green-Kubo formula
    # 8.61673324e-5 is Boltzmann constant in eV/K
    # Multiply by variance of J to scale normalized ACF
    scale = 1 / (V * T**2) / (8.61673324*10**(-5)) * J.var()
    
    # Conversion factor from eV/(ps·Å²·K) to W/mK
    metal2SI = 1.602176634*10**3
    
    # Compute kappa using only A1, A2, tau1, tau2
    kappa_C = scale * (A1 * tau1 + A2 * tau2) * metal2SI
    
    # Compute kappa including Y0 contribution times tau_FA
    kappa_F = scale * (A1 * tau1 + A2 * tau2 + Y0 * tau_FA) * metal2SI
    
    # Return both conductivity values
    return kappa_C, kappa_F

# ================================================================================================== #

# "Advanced" methods

def plot_data(filenames, path, dir, labels, cols, timesteps, segments=None, max_points=50000):
    """
    Plot time series of multiple columns from multiple simulation files, aligning them in time.

    Parameters:
        filenames : list of str
            Names of CSV files containing the data.
        path : str
            Path where files are located.
        dir : str
            Directory to save the plots and log.
        labels : list of str
            Labels for each file in the plot legend.
        cols : list of str
            Columns to plot from the files.
        timesteps : list of float
            Time step corresponding to each file (ps per step).
        segments : list of int, optional
            Groups of files; new segment → offset in time.
        max_points : int
            Maximum number of points to plot (downsampling if necessary).
    """

    # If segments not provided, assign each file to its own segment
    if segments is None:
        segments = list(range(1, len(filenames)+1))

    # Create a subplot grid: 2 rows, enough columns for all `cols`
    fig, axes = plt.subplots(2, int(np.ceil(len(cols)/2)), figsize=(12, 2*len(cols)//2))
    axes = axes.flatten()  # Flatten for easy indexing

    # Create output directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)  
    
    # --- Compute cumulative time offsets to align simulations ---
    offsets = [0.0]  # first file has zero offset
    for k in range(1, len(filenames)):
        prev_df = pd.read_csv(path + filenames[k-1], sep=r"\s+", comment="#",
                              names=["TimeStep"] + cols,
                              usecols=["TimeStep"])
        duration = prev_df["TimeStep"].iloc[-1] * timesteps[k-1]

        if segments[k] == segments[k-1]:
            # same group → overlapping time
            offsets.append(offsets[-1])  
        else:
            # new group → concatenate in time
            offsets.append(offsets[-1] + duration)
            
    # Open log file to write summary statistics
    with open(dir + "/therm.txt", "w") as log:
        for i, col in enumerate(cols):
            for fname, label, timestep in zip(filenames, labels, timesteps):
                # --- Read file in chunks for large datasets ---
                df_list = []
                for chunk in pd.read_csv(path+fname, sep=r"\s+", comment="#",
                                         names=["TimeStep"] + cols + ["Jx","Jy","Jz"],
                                         chunksize=10**6):
                    df_list.append(chunk[["TimeStep", col]])
                df = pd.concat(df_list)
                
                # --- Downsample if too many points ---
                if len(df) > max_points:
                    stride = len(df) // max_points
                    df = df.iloc[::stride]
                
                # Time axis with offset for alignment
                x = df["TimeStep"].values * timestep + offsets[filenames.index(fname)]
                y = df[col].values
                
                # Plot column vs time
                axes[i].plot(x, y, label=label)
                
                # Compute statistics only for last file
                mean_val = 0
                std_val = 0
                if fname == filenames[-1]:
                    mean_val = np.mean(y)
                    std_val = np.std(y, ddof=1)
                    err_val = std_val / np.sqrt(len(y))
                    # Linear regression to check drift
                    slope, intercept, *_ = linregress(x, y)
                    durata = x[-1] - x[0]
                    drift_totale = slope * durata

                    # Print statistics to log depending on column type
                    if col=="V":
                        print(f"{fname:<9s} | {col:5s} -> media={mean_val:.4f}, err={err_val:.4e}", file=log)
                    if col=="Etot":
                        drift_perc = abs(drift_totale/mean_val*100)
                        drift_str = "negligible" if (drift_perc < 0.1/100 or drift_totale==0) else "significant"
                        print(f"{fname:<9s} | {col:5s} -> mean={mean_val:.4f}, err={err_val:.4e}, drift%={drift_perc:.4e}%, drift={drift_totale:.4e} ({drift_str})", file=log)
                    if col != "Etot" and col != "V":
                        drift_str = "negligible" if (abs(drift_totale) < 0.1 * std_val or drift_totale==0) else "significant"
                        print(f"{fname:<9s} | {col:5s} -> mean={mean_val:.4f}, err={err_val:.4e}, std={std_val:.4e}, drift={drift_totale:.4e} ({drift_str})", file=log)
                        # Plot regression line to show drift
                        axes[i].plot(x, intercept + slope*x, "--", alpha=0.7, zorder=6)
                
            # Set axis labels, title, grid, legend
            axes[i].set_xlabel("Time [ps]")
            axes[i].set_ylabel(col)
            axes[i].grid(True)
            axes[i].set_title(f"Evolution of {col}")
            axes[i].legend(loc="best")
            
            # Special y-axis limits for V and other columns
            if fname == filenames[-1]:
                if col=="V":
                    y_min = mean_val * 0.99998
                    y_max = mean_val * 1.00002
                else:
                    y_min = mean_val - 8 * std_val
                    y_max = mean_val + 8 * std_val
                axes[i].set_ylim(y_min, y_max)
        
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(dir+"/therm.pdf", dpi=300)


def analyze_kappa(file_path, dt=0.001, direction='x', window_size=100000, step_size=100000,
                  plot_acf_flag=False, plot_EF_flag=False, plot_kappa_flag=False,
                  out_dir="prod"):
    """
    Analyze thermal conductivity from a LAMMPS dump file using sliding windows.
    
    Parameters:
        file_path : str
            Path to the LAMMPS output file containing heat flux, temperature, etc.
        dt : float
            Time step between points in ps.
        direction : str
            Heat flux direction to analyze ('x', 'y', 'z').
        window_size : int
            Number of points in each sliding window.
        step_size : int
            Step size to slide the window.
        plot_acf_flag : bool
            Whether to plot autocorrelation and its fit.
        plot_EF_flag : bool
            Whether to plot fluctuation F(t) and mean E(t).
        plot_kappa_flag : bool
            Whether to plot kappa(t) with FA and FD markers.
        out_dir : str
            Directory to save results (.npz files).
    """
    
    print(f"Analyzing thermal conductivity in direction '{direction}'")
    
    # Create output directories if they do not exist
    dir_npz = os.path.join(out_dir, direction, "npz")
    os.makedirs(dir_npz, exist_ok=True)
    
    # Count total lines in file to determine number of points
    n_lines = sum(1 for _ in open(file_path))
    n_points = n_lines
    
    # --- Sliding window loop ---
    for start in tqdm(range(0, n_points - window_size + 1, step_size), desc="Sliding windows"):
        end = start + window_size
        
        # Load only the current block from file using pandas
        df = pd.read_csv(
            file_path, sep=r"\s+", comment="#", header=None,
            names=["step", "T", "KE", "PE", "Etot", "P", "V", "Jx", "Jy", "Jz"],
            skiprows=start + 2, nrows=window_size
        )
        
        df = df.dropna()
        if len(df) < window_size:
            # Skip incomplete blocks
            print(f"Block {start}-{end} incomplete, skipping.")
            continue
        
        # --- Extract main quantities ---
        J = df[f"J{direction}"].values   # Heat flux in selected direction
        T = df["T"].mean()               # Average temperature
        V = df["V"].mean()               # Average volume
        t = np.arange(window_size) * dt  # Time axis
        
        # --- Compute autocorrelation and integrate ---
        ac = autocorrelation_fft(J)      # Compute ACF
        kk = integrate_acf(ac)           # Integrate ACF
        kap = compute_kappa(kk, J, V, T) # Compute thermal conductivity
        
        # --- Fit ACF with double exponential ---
        pars = fit_acf(t, ac)
        ac_fit = ACF_fit(t, *pars)
        if plot_acf_flag:
            plot_acf(t, ac, ac_fit)
        
        # Compute integrated kappa from fitted ACF
        kk_fit = integrate_acf(ac_fit)
        kap_fit = compute_kappa(kk_fit, J, V, T)
        
        # --- Compute fluctuation F(t) and mean E(t) ---
        F, E, first_peak_x, first_zero_x = compute_F_E(ac, J, t)
        if plot_EF_flag:
            plot_F_E(t, F, first_peak_x, E, first_zero_x, n_plot=100000)
        
        # --- Compute FA and FD thermal conductivities ---
        tau_FA, kappa_FA, tau_FD, kappa_FD = compute_kappa_FA_FD(
            t, kap, first_peak_x, first_zero_x, timestep=dt
        )
        if plot_kappa_flag:
            plot_kappa_FA_FD(t, kap, kap_fit, tau_FA, kappa_FA, tau_FD, kappa_FD, n_plot=100000)
        
        # --- Compute thermal conductivity using Chen et al method ---
        kappa_C, kappa_F = compute_kappa_Chen(pars, J, V, T, tau_FA)
        
        # --- Save results in compressed .npz file ---
        np.savez_compressed(
            f"{dir_npz}/ws{window_size}_ss{step_size}_start{start}.npz",
            start=start, end=end,
            tau_FA=tau_FA, kappa_FA=kappa_FA,
            tau_FD=tau_FD, kappa_FD=kappa_FD,
            kap=kap, kapfit=kap_fit, pars=pars,
            kappa_C=kappa_C, kappa_F=kappa_F
        )
        
        # --- Free memory for next iteration ---
        del df, J, ac, ac_fit, kap, kap_fit
        gc.collect()


def plot_kappas(base_dir="prod", direction='x', timestep=0.001, n_points=100000):
    """
    Plot and summarize thermal conductivity κ(t) from all .npz files in a given direction.
    Generates scatter plots for FA/FD points and saves mean/std values to a text file.
    
    Parameters:
        base_dir : str
            Base directory containing results.
        direction : str
            Direction to analyze ('x', 'y', 'z').
        timestep : float
            Time step between consecutive points in ps.
        n_points : int
            Maximum number of points to consider from each file.
    """
    
    # Directory containing .npz results
    directory = os.path.join(base_dir, direction, "npz")
    npz_files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in '{directory}'")
        return

    # Initialize lists to collect all results
    all_kap = []       # κ(t) curves
    all_tau_FA = []    # FA times
    all_kappa_FA = []  # FA κ
    all_tau_FD = []    # FD times
    all_kappa_FD = []  # FD κ
    all_kappa_C = []   # Chen et al κ_C
    all_kappa_F = []   # Chen et al κ_F

    # Loop over .npz files
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        # Skip files missing required data
        if "kap" not in data or "tau_FA" not in data or "tau_FD" not in data:
            print(f"File {f} does not contain valid data, skipping.")
            continue

        # Append κ(t) and characteristic points to lists
        kap = data["kap"][:n_points]
        all_kap.append(kap)
        all_tau_FA.append(float(data["tau_FA"]))
        all_kappa_FA.append(float(data["kappa_FA"]))
        all_tau_FD.append(float(data["tau_FD"]))
        all_kappa_FD.append(float(data["kappa_FD"]))
        # Filter reasonable Chen et al κ values
        if 0.0 < float(data["kappa_C"]) < 10.0:
            all_kappa_C.append(float(data["kappa_C"]))
        if 0.0 < float(data["kappa_F"]) < 10.0:
            all_kappa_F.append(float(data["kappa_F"]))

    if not all_kap:
        print("No valid data found.")
        return

    # Convert lists to numpy arrays for analysis
    kap_matrix = np.vstack(all_kap)
    kap_mean = np.mean(kap_matrix, axis=0)
    kap_std = np.std(kap_matrix, axis=0)
    taus_FA = np.array(all_tau_FA)
    kappas_FA = np.array(all_kappa_FA)
    taus_FD = np.array(all_tau_FD)
    kappas_FD = np.array(all_kappa_FD)
    kappas_C = np.array(all_kappa_C) if all_kappa_C else None
    kappas_Fit = np.array(all_kappa_F) if all_kappa_F else None

    # Time axis
    x = np.arange(len(kap_mean)) * timestep

    # --- Plot κ(t) curves ---
    plt.figure(figsize=(9,6))
    for kap in all_kap:
        plt.plot(x, kap, alpha=0.3, lw=1)  # Individual curves
    
    # Mean κ(t) curve with shaded ±1σ
    plt.plot(x, kap_mean, color='blue', lw=2, label='Mean κ(t)')
    plt.fill_between(x, kap_mean - kap_std, kap_mean + kap_std, color='blue', alpha=0.2, label=r'±1$\sigma$')
    
    # Scatter FA points
    plt.scatter(taus_FA, kappas_FA, color='red', edgecolors='black', s=20, label='FA points')
    # Highlight mean FA
    plt.scatter(np.mean(taus_FA), np.mean(kappas_FA), color='gold', s=200, marker='*', edgecolors='black',
                label=f'Mean FA (τ={np.mean(taus_FA):.2f}, κ={np.mean(kappas_FA):.2f})')
    
    # Scatter FD points
    plt.scatter(taus_FD, kappas_FD, color='green', edgecolors='black', s=20, label='FD points')
    # Highlight mean FD
    plt.scatter(np.mean(taus_FD), np.mean(kappas_FD), color='lime', s=200, marker='*', edgecolors='black',
                label=f'Mean FD (τ={np.mean(taus_FD):.2f}, κ={np.mean(kappas_FD):.2f})')
    
    # Labels, limits, grid, and legend
    plt.xlabel('Correlation time (ps)')
    plt.ylabel('Thermal conductivity κ (W/m·K)')
    plt.xlim(-5, n_points*timestep)
    plt.legend(loc='upper left')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    
    # Add text box with direction info
    plt.text(0.95, 0.05,                 #x=95% from left, y=5% from bottom → lower-right corner
             "Direction: "+direction,
             fontsize=12,
             fontweight='bold',
             color='black',
             ha='right',                 # horizontal alignment
             va='bottom',                # vertical alignment
             bbox=dict(
                 boxstyle="round,pad=0.3",
                 facecolor="white",
                 edgecolor="black",
                 linewidth=1
             ),
             transform=plt.gca().transAxes  # use axes coordinates
             )
    
    plt.tight_layout()
    
    # Save figure
    outdirectory = os.path.join(base_dir, direction)
    plt.savefig(os.path.join(outdirectory, f"kappa_summary_{direction}.pdf"), dpi=300)
    plt.close()

    # --- Write summary statistics to text file ---
    out_file = f"kappa_summary_{direction}.txt"
    with open(os.path.join(outdirectory, out_file), 'w') as f:
        f.write("FA, tau_FA_mean, tau_FA_std, kappa_FA_mean, kappa_FA_std\n")
        f.write(f"{np.mean(taus_FA):.4f},{np.std(taus_FA):.4f},{np.mean(kappas_FA):.4f},{np.std(kappas_FA):.4f}\n")
        f.write("FD, tau_FD_mean, tau_FD_std, kappa_FD_mean, kappa_FD_std\n")
        f.write(f"{np.mean(taus_FD):.4f},{np.std(taus_FD):.4f},{np.mean(kappas_FD):.4f},{np.std(kappas_FD):.4f}\n")
        if kappas_C is not None:
            f.write("Chen et al., kappa_C_mean, kappa_C_std, kappa_F_mean, kappa_F_std\n")
            f.write(f"{np.mean(kappas_C):.4f},{np.std(kappas_C):.4f},{np.mean(kappas_Fit):.4f},{np.std(kappas_Fit):.4f}\n")
    
    print(f"Mean values and standard deviations written to: {os.path.join(outdirectory, out_file)}")


# by Antonio Sacco & commented by ChatGPT

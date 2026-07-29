Folder containing scripts for designing, optimizing, fitting, and evaluating regularized windows (Hann combined with double exponential envelopes) and their SSM rational fraction approximations.
The scripts are:

### Compare_to_hann
Compare a regularized double-exponential window against a standard Hann window in terms of time shape, Overlap-Add (OLA) error, and frequency attenuation.

### Compute_and_plot_rational_fit
Compute and plot the 3-state SSM rational fraction fit (sum of 3 complex low-pass filters) for a regularized window magnitude spectrum.

### Find_optimal_parameters
Perform a grid search over window parameters ($\epsilon_1$, $\lambda_1$, $\lambda_2$) to minimize OLA modulation error and spectral gradient oscillations.

### Plot_compare_configs
Compare and visualize 3-state SSM rational fraction fits and magnitude spectra across multiple regularized window configurations.

### Plot_regularized_windows
Parse `configs.txt` and plot time-domain profiles and frequency attenuation spectra for regularized windows across OLA error targets.

### Plot_single_window
Plot detailed time-domain shape, OLA modulation error, and frequency attenuation spectrum for a single window configuration.

### Result_store
Define data structures (`ResultsStore`, `OptimalParams`) to store, serialize to JSON, and query window regularization search results.

### Sweep_window_parameters
Perform a systematic grid search sweep across hop factors and OLA penalty weights, saving optimal window parameters to JSON via `ResultsStore`.

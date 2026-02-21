
This is a professional-grade Battery SOC Estimation Tool designed for researchers and engineers working with lithium-ion batteries (specifically NMC Chemistry in this case).


🔋 Battery State-of-Charge (SOC) Estimation Tool
OCV Modeling & 2RC-Hysteresis Extended Kalman Filter (EKF)
This repository provides an end-to-end Python-based tool for accurate battery state estimation using laboratory test data (Neware/Arbin Excel exports). It combines physics-based modeling with advanced signal processing to provide robust SOC tracking even under dynamic loading conditions.

🚀 Key Features
1. Robust OCV-SOC Characterization
Instead of relying on hardware-calculated SOC columns (which are often prone to drift), the tool implements a Coulomb Counting (Ah Integration) Engine .

It automatically parses OCV (Open Circuit Voltage) test files.

It extracts voltage "tail" points during rest periods.

It generates a high-fidelity OCV lookup table using median-filtered data bins.

2. Advanced 2-RC + Hysteresis Model
The Estimation engine uses a second-order Thevenin equivalent circuit model (2-RC) to Capture complex battery dynamics:

R0: Ohmic resistance (instantaneous voltage drop).

R1/C1 & R2/C2: Short-term and long-term diffusion/polarization effects (transient voltage recovery).

Hysteresis (k_h): Captures the voltage difference between charging and discharging states at the same SOC.

3. Extended Kalman Filter (EKF)
The core of the app is a customized EKF algorithm that solves the "sensor fusion" problem by balancing two inputs:

The Model: Predicts SOC based on current integration (Amperes over time).

The Measurement: Corrects the Prediction by comparing the Predicted terminal voltage against the actual measured voltage.

Numerical Stability: Includes a symmetry constraint on the covariance matrix to prevent divergence during long-duration tests.

4. Interactive Visualization
Built with Streamlit , the tool provides real-time feedback:

Voltage Fit: Visual comparison of EKF-predicted voltage vs. measured voltage (RMSE calculation).

SOC Validation: Comparison against multiple reference modes (Capacity-based, OCV-inversion, or hardware labels).

Residual Analysis: Tracks "Rest Residuals" to help tune R and C parameters.

🛠️ How to Use
Upload: Provide your OCV and HPPC (Hybrid Pulse Power Characterization) Excel files.

Configure: Set your cell capacity (Q_AH) and initial EKF guesses (R, C, and Noise values).

Run: The app processes the time-series data and generates the SOC curve.

Export: Download the processed data as a CSV for further reporting.

📊 Technical Stack
Language: Python 3.9+

UI: Streams

Processing: NumPy, Pandas, SciPy (Interpolation)

Plotting: Matplotlib


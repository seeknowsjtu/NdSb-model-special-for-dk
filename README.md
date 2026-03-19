# NdSb3Tmodel: Non-equilibrium Dynamics Simulator for NdSb

**NdSb3Tmodel** is a highly specialized, physics-informed simulation and fitting suite designed to model the ultrafast non-equilibrium dynamics of Neodymium Antimonide (NdSb). 

Traditional Three-Temperature Models (3TM) often fail to capture complex magnetic phase transitions. This tool extends the standard 3TM by explicitly coupling the thermodynamic baths (Electron $T_e$, Spin $T_s$, Lattice $T_l$) with phenomenological order parameters: the primary magnetic order parameter ($m$) and a secondary spin-reorientation parameter ($\eta$). 

It is tailored for analyzing time-resolved pump-probe spectroscopy and tr-ARPES data, featuring an interactive GUI for real-time simulation, parameter tuning, and bounded least-squares fitting.

## ✨ Key Features

* **Extended 3TM ODE Solver**: Solves a coupled system for $T_e$, $T_s$, $T_l$, $m$, and $\eta$.
* **Advanced Thermodynamics**: 
  * Integrates Debye lattice heat capacity (optimized via cached PCHIP interpolation).
  * Includes multi-level Schottky contributions from Crystal Electric Field (CEF) states.
  * Implements Linear Spin-Wave Theory (LSWT) for magnon heat capacity across the Brillouin zone.
* **State-Dependent Couplings**: Effective energy transfer channels ($G_{es}, G_{el}, G_{sl}$) dynamically respond to the magnetic order $m$ and reorientation $\eta$, including critical enhancements near $T_N$ (15 K) and $T_R$ (13 K).
* **Interactive GUI**: A built-in Tkinter/Matplotlib dashboard allows users to adjust pump parameters, thermodynamic anchors, and couplings to immediately preview transient dynamics.
* **Robust Fitting Engine**: Built on `scipy.optimize.least_squares` with `soft_l1` loss, allowing simultaneous or independent fitting of electron temperature $T_e(t)$ and ARPES spectral weight $S(t)$ against experimental CSV data.

## 📂 Project Structure

* `main.py`: Application entry point. Hosts the Tkinter GUI, dynamic plotting, and the headless CLI fallback.
* `solver.py`: The core ODE engine (`NdSb3TM`). Handles the Right-Hand Side (RHS) equations, pump profiling, and rigorous energy-balance diagnostics.
* `physics_engine.py`: Thermodynamics and magnon models. Includes the Debye integral cache, CEF Schottky calculations, and phenomenological coupling scaling rules.
* `data_io.py`: Handles CSV parsing with auto-column detection and the parameter-packing logic for the least-squares fitting algorithm.
* `config.py`: Central repository for physical constants, NdSb material parameters, and default simulation variables.
* `gui_component.py`: Custom scrollable Tkinter frame utility for the extensive parameter interface.

## ⚙️ Installation & Requirements

The project requires Python 3.8+ and relies on standard scientific libraries. 

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/seeknowsjtu/NdSb3Tmodel.git](https://github.com/seeknowsjtu/NdSb3Tmodel.git)
   cd NdSb3Tmodel
Install dependencies:Bashpip install numpy scipy matplotlib
(Note: The GUI requires tkinter, which is included in most standard Python distributions. If you are on Linux, you may need to install it via your package manager, e.g., sudo apt-get install python3-tk.)🚀 UsageLaunching the GUISimply run the main script. The GUI will open automatically if your environment supports it.Bashpython main.py
Fitting Experimental DataTo fit experimental data, your CSV file should contain a header row. The tool auto-detects column names based on common conventions:Time: t, time_ps, tpsElectron Temp: Te, Te_K, temp_eSpectral Weight: S, sw, intensityWorkflow in GUI:Click Load CSV... and select your experimental data.In the left panel, adjust the starting parameters (guess) to roughly match your curves.Click Fit Te, Fit S, or Fit Te + S.Review the fitted curves. Click Apply Fit → Params to lock them in.Headless/CLI ModeIf tkinter is not available (e.g., on a remote SSH server), running python main.py will automatically fall back to a headless demonstration, running a default pulse simulation and printing the maximum transient temperatures to the console.🔬 Physics ParametersThe model defaults are anchored to the physical properties of NdSb:$T_N$ (Néel Temperature): 15.0 K$T_R$ (Spin-reorientation): 13.0 K$\Theta_D$ (Debye Temp): 200.0 KMagnetic gaps and CEF levels are pre-configured in config.py but remain fully adjustable within the GUI.

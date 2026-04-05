# Antenna AI Predictor 🚀

An AI-powered system designed to revolutionize antenna design by replacing slow electromagnetic (EM) simulations with near-instantaneous neural network predictions. This project features a **Forward Predictor** (Dimensions → Performance) and an **Inverse Encoder** (Performance Goals → Dimensions).

## 🌟 Overview

Traditional antenna design involves iterative EM simulations that can take minutes or hours per iteration. This system uses **Deep Learning (PyTorch)** to:
1.  **Predict Performance:** Instantly estimate Bandwidth (GHz) and Gain (dBi) for given physical dimensions.
2.  **Inverse Design:** Provide the required physical dimensions (Length, Width, Feed Position) to achieve specific performance targets.

## 🛠️ Software & Tools Required

To run this project, you need the following installed on your system:

### 1. Python Environment
- **Python 3.8+** (Recommended: 3.10 or 3.11)
- **PyTorch:** For building and training the Neural Networks.
- **Pandas & NumPy:** For data handling and numerical operations.
- **Scikit-Learn:** For data scaling and preprocessing.
- **Matplotlib:** For visualizing results and performance curves.

### 2. EM Simulation (Data Generation)
- **OpenEMS:** An open-source electromagnetic field solver based on the FDTD method.
  - *Note:* The scripts are configured to use OpenEMS located at `C:\Users\sri\Downloads\openEMS_x64_v0.0.36-93-g7b9cd51_msvc\openEMS`.
- **CSXCAD:** For geometry description within OpenEMS.

## 📂 Project Structure

- `generate_dataset_openems.py`: Interfaces with OpenEMS to run automated EM simulations and generate a training dataset (`dataset_patch.csv`).
- `train_predictor.py`: The core AI engine. Defines, trains, and validates the `AntennaPredictor` (Forward) and `AntennaEncoder` (Inverse) models.
- `visualize_results.py`: Generates plots to compare AI predictions against ground truth data.
- `dataset.csv` / `dataset_patch.csv`: The simulated data containing antenna dimensions and their corresponding performance metrics.

## 🚀 How to Use

### Step 1: Install Dependencies
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Step 2: Generate Data (Optional)
If you want to create a new dataset using OpenEMS:
```bash
python generate_dataset_openems.py
```
*Ensure OpenEMS is correctly installed and the path is updated in the script.*

### Step 3: Train the AI Models
Run the training script to teach the neural networks how antennas behave:
```bash
python train_predictor.py
```
This will output the training loss and a "Speed & Accuracy Proof" showing how the AI performs compared to traditional methods.

### Step 4: Visualize Performance
```bash
python visualize_results.py
```

## 📊 AI System Architecture

- **Predictor (Forward Model):**
  - Input: `[Length, Width, FeedPosition]`
  - Output: `[Bandwidth, Gain]`
  - Architecture: Multi-layer Perceptron (128 -> 256 -> 128 units).

- **Encoder (Inverse Model):**
  - Input: `[Target Bandwidth, Target Gain]`
  - Output: `[Required Length, Width, FeedPosition]`
  - Architecture: Multi-layer Perceptron (128 -> 256 -> 128 units).

## 📈 Key Benefits
- **Speed:** Predictions in milliseconds vs. minutes for EM solvers.
- **Optimization:** Quickly iterate through thousands of designs to find the global optimum.
- **Automation:** End-to-end flow from performance goals to physical dimensions.

---
*Developed by Srikanth Chowdary*

# This script is a direct adaptation of the code from the official repository:
# https://github.com/ubicomplab/oximetry-phone-cam-data
#
# The goal is to preserve the original structure as much as possible, with minimal changes
# to allow processing of a new video file instead of the provided .mat dataset.
#
# Main Modifications:
# 1. The PPG_Data.__init__ method is modified to accept a video file path and extract
#    signals using OpenCV.
# 2. A separate SpO2Estimator class has been added to implement the classical
#    "ratio-of-ratios" algorithm, since the original deep learning model is not available.
# 3. Added sliding window calculation for time-series SpO2/HR and Excel export via pandas.
#
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd  # Newly added: Import pandas library for Excel file generation and saving later

class PPG_Data:
    """
    This class is a direct adaptation of the PPG_Data class from the official repository's data_loader.py.
    It's modified to load data from a video file instead of a pre-processed .mat file.
    """
    def __init__(self, video_path, sig_length=90):
        self.sig_length = sig_length
        self.video_path = video_path
        # --- MODIFICATION START ---
        # The original code loaded a .mat file here.
        # We replace that logic with a function to extract signals from a video.
        self._load_from_video()
        # --- MODIFICATION END ---

    def _load_from_video(self):
        """
        A new private method to handle video processing. It populates the same
        instance variables (self.wave, self.time, etc.) as the original code.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        
        self.fs = cap.get(cv2.CAP_PROP_FPS)
        if self.fs == 0:
            print("Warning: FPS is 0. Defaulting to 30.")
            self.fs = 30

        # The 'wave' variable in the original .mat file stores R, G, B signals.
        # We will create it from the video.
        r_signal, g_signal, b_signal = [], [], []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r_signal.append(np.mean(frame_rgb[:, :, 0]))
            g_signal.append(np.mean(frame_rgb[:, :, 1]))
            b_signal.append(np.mean(frame_rgb[:, :, 2]))
        
        cap.release()
        # The original 'wave' is a (3, N) numpy array. We replicate that structure.
        self.wave = np.array([r_signal, g_signal, b_signal])
        
        # Create a time vector, similar to the original data structure.
        self.time = np.arange(self.wave.shape[1]) / self.fs
        
        # Ground truth SpO2 is not available in a new video.
        self.gt_spo2 = np.array([])
        # Subject ID is not applicable.
        self.sub_id = "N/A"

    def __len__(self):
        # This remains unchanged from the original repository code.
        return self.wave.shape[1] - self.sig_length + 1

    def __getitem__(self, idx):
        # This remains unchanged, returning a window of the signal.
        return self.wave[:, idx:idx + self.sig_length]

class SpO2Estimator:
    """
    This is a supplementary class to provide the SpO2 estimation logic,
    which replaces the unavailable deep learning model from the paper.
    """
    def __init__(self, wave, fs):
        self.wave = wave  # Expects a (3, N) numpy array [R, G, B]
        self.fs = fs

    def estimate(self):
        r_signal = self.wave[0, :]
        b_signal = self.wave[2, :]
        # 1. Filter signals to get AC components
        nyquist = 0.5 * self.fs
        low = 0.67 / nyquist
        high = 4.0 / nyquist
        b, a = butter(1, [low, high], btype='band')
        r_ac = filtfilt(b, a, r_signal)
        b_ac = filtfilt(b, a, b_signal)

        # 2. Get DC components
        r_dc = np.mean(r_signal)
        b_dc = np.mean(b_signal)

        # 3. Calculate "Ratio of Ratios"
        R = (np.std(r_ac) / r_dc) / (np.std(b_ac) / b_dc)
        # 4. Apply empirical formula
        spo2 = 110 - 25 * R
        spo2 = max(80, min(99.9, spo2))  # Clamp to plausible range

        # 5. Estimate heart rate for reference
        peaks, _ = find_peaks(r_ac, height=np.std(r_ac), distance=self.fs/4)
        if len(peaks) < 2:
            hr = "N/A"
        else:
            hr = 60.0 / (np.mean(np.diff(peaks)) / self.fs)
            
        return spo2, hr

    def estimate_sliding(self, window_size=3):
        """
        Newly added: Sliding window calculation method, used to generate SpO2 and heart rate data at each time point
        window_size: Sliding window size (unit: seconds), default is 3 seconds (balances calculation accuracy and result stability)
        Returns: Numpy arrays of SpO2 values and heart rate values at each time point
        """
        window_samples = int(window_size * self.fs)  # Newly added: Convert window time to number of samples (adapts to video FPS)
        total_samples = self.wave.shape[1]  # Newly added: Get total number of signal samples (i.e., total video frames)
        spo2_list = []  # Newly added: List to store SpO2 results at each time point
        hr_list = []    # Newly added: List to store heart rate results at each time point
        
        # Newly added: Iterate over each time point (current point as window end), extract signals from previous 'window_size' seconds
        for end_idx in range(window_samples, total_samples + 1):
            start_idx = end_idx - window_samples  # Newly added: Calculate start index of the sliding window
            # Newly added: Extract red (R) and blue (B) channel signals in current window (SpO2 calculation relies on these two channels)
            r_win = self.wave[0, start_idx:end_idx]
            b_win = self.wave[2, start_idx:end_idx]
            
            # Newly added: Reuse signal processing logic from original 'estimate' method to calculate AC/DC components in window
            nyquist = 0.5 * self.fs
            low = 0.67 / nyquist
            high = 4.0 / nyquist
            b, a = butter(1, [low, high], btype='band')
            r_ac = filtfilt(b, a, r_win)
            b_ac = filtfilt(b, a, b_win)
            r_dc = np.mean(r_win)
            b_dc = np.mean(b_win)
            
            # Newly added: Calculate SpO2 in current window, add zero-division check to avoid errors
            if r_dc == 0 or b_dc == 0:
                spo2 = np.nan  # Newly added: Mark invalid data as NaN (will be filtered automatically in Excel later)
            else:
                R = (np.std(r_ac) / r_dc) / (np.std(b_ac) / b_dc)
                spo2 = 110 - 25 * R
                spo2 = max(80, min(99.9, spo2))  # Newly added: Limit SpO2 to reasonable physiological range (80%-99.9%)
            
            # Newly added: Calculate heart rate in current window, mark as invalid if fewer than 2 peaks exist
            peaks, _ = find_peaks(r_ac, height=np.std(r_ac), distance=self.fs/4)
            if len(peaks) < 2:
                hr = np.nan  # Newly added: Mark invalid heart rate as NaN
            else:
                hr = 60.0 / (np.mean(np.diff(peaks)) / self.fs)  # Newly added: Calculate heart rate via peak interval
        
            spo2_list.append(spo2)  # Newly added: Add current window's SpO2 result to the list
            hr_list.append(hr)      # Newly added: Add current window's heart rate result to the list
        
        # Newly added: Fill NaN for the first (window_size - 1) seconds (insufficient window data for initial points)
        pad_nan = [np.nan] * (window_samples - 1)
        spo2_list = pad_nan + spo2_list
        hr_list = pad_nan + hr_list
        
        return np.array(spo2_list), np.array(hr_list)  # Newly added: Convert lists to numpy arrays and return

def main():
    # --- User Configuration ---
    # IMPORTANT: Replace this with the path to your fingertip video file.
    video_path = "sample/finger_video.mp4"
    window_size = 3  # Newly added: Sliding window size configuration (unit: seconds), adjustable (2-4 seconds recommended)
    # --------------------------

    # 1. Load data using the adapted PPG_Data class (faithful to data_loader.py)
    print(f"Loading data from video: {video_path}")
    ppg_data = PPG_Data(video_path=video_path)
    print(f"Data loaded. Found {ppg_data.wave.shape[1]} frames at {ppg_data.fs:.2f} FPS.")

    # 2. Newly added: Call sliding window method to calculate SpO2 and heart rate at each time point
    estimator = SpO2Estimator(ppg_data.wave, ppg_data.fs)
    spo2_time_series, hr_time_series = estimator.estimate_sliding(window_size=window_size)
    
    # 3. Newly added: Construct Excel data structure and save the file
    # Newly added: Create DataFrame (table structure) with 3 columns: Time, SpO2, Heart Rate
    excel_data = pd.DataFrame({
        "Time (s)": ppg_data.time,  # Newly added: Time series corresponding to video frames (matches original signal time)
        "Estimated SpO2 (%)": spo2_time_series,  # Newly added: Time-series SpO2 results
        "Estimated Heart Rate (BPM)": hr_time_series  # Newly added: Time-series heart rate results
    })
    # Newly added: Define Excel filename, filter rows with invalid values (NaN) before saving (remove 'dropna' to keep all data)
    excel_filename = "spo2_hr_time_series.xlsx"
    excel_data.dropna(subset=["Estimated SpO2 (%)", "Estimated Heart Rate (BPM)"]).to_excel(
        excel_filename, index=False  # Newly added: 'index=False' means not saving table row numbers
    )
    print(f"\nTime-series data saved to {excel_filename}")  # Newly added: Print success message for Excel saving

    # 4. Newly added: Calculate and print overall statistics (for quick data overview)
    valid_spo2 = spo2_time_series[~np.isnan(spo2_time_series)]  # Newly added: Filter valid SpO2 data
    valid_hr = hr_time_series[~np.isnan(hr_time_series)]        # Newly added: Filter valid heart rate data
    print("\n--- Overall Estimation Statistics ---")
    if len(valid_spo2) > 0:
        print(f"Average SpO2: {np.mean(valid_spo2):.2f} %")  # Newly added: Print average SpO2
    else:
        print("Average SpO2: No valid data")
    if len(valid_hr) > 0:
        print(f"Average Heart Rate: {np.mean(valid_hr):.2f} BPM")  # Newly added: Print average heart rate
    else:
        print("Average Heart Rate: No valid data")
    print("-------------------------------------\n")

    # 5. Visualize the signals (faithful to visualize.py)
    print("Generating PPG signal plot...")
    r, g, b = ppg_data.wave[0], ppg_data.wave[1], ppg_data.wave[2]
    
    plt.figure(figsize=(15, 5))
    plt.plot(ppg_data.time, r, 'r', label='Red Channel')
    plt.plot(ppg_data.time, g, 'g', label='Green Channel')
    plt.plot(ppg_data.time, b, 'b', label='Blue Channel')
    plt.title('Extracted PPG Signals from Video')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Pixel Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig("ppg_signals_from_video.png")
    print("PPG signal plot saved to ppg_signals_from_video.png")

if __name__ == "__main__":
    main()

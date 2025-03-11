import numpy as np
class cfg:
    def __init__(self, peak0, range_ns, time_bin, time_integration_ms, low_coinc_windows, high_coinc_windows):
        self.peak0 = peak0
        self.range_ns = range_ns
        self.time_bin = time_bin
        self.time_integration_ms = time_integration_ms
        self.low_coinc_windows = low_coinc_windows
        self.high_coinc_windows = high_coinc_windows


def calculate_coincidences(config, timestamps1, timestamps2):
        """
        Calculate coincidences between two timestamp arrays.

        Args:
            wr_time (String): The White rabbit time identifying the two timestamps
            timestamps1 (array-like): Array of timestamps for the first channel.
            timestamps2 (array-like): Array of timestamps for the second channel.

        Returns:
            dict: A dictionary containing:
            - histogram values, bin edges
            - peak time
            - low coincidence window bounds
            - coincidence rate in the given window
            - rates for timestamps1 and timestamps2
        """
        # Initialize variables
        timestamps2 = timestamps2 + config.peak0
        range_ps = 1e3 * config.range_ns  # Convert range from ns to ps
        half_range_ns = config.range_ns /2
        half_range_ps = range_ps / 2
        bins_n = int(config.range_ns/config.time_bin)
        integration_time_s = config.time_integration_ms / 1000 if config.time_integration_ms else 1

        # Find coincidences using a sliding window approach
        coincidences = []
        idx2 = 0
        for t1 in timestamps1:
            while idx2 < len(timestamps2) and timestamps2[idx2] < t1 - half_range_ps:
                idx2 += 1

            start_idx = idx2
            while idx2 < len(timestamps2) and timestamps2[idx2] <= t1 + half_range_ps:
                dtime = t1 - timestamps2[idx2]
                coincidences.append(dtime)
                idx2 += 1
            # Reset idx2 to the start index for the next iteration
            idx2 = start_idx
        # Convert coincidences to a numpy array
        coincidences = np.array(coincidences)
        # Calculate the histogram of time differences
        histo_vals, bin_edges = np.histogram(coincidences, bins=bins_n, range=(-half_range_ps, half_range_ps))
        # Find the bin with the peak (maximum count)
        peak_index = np.argmax(histo_vals)
        peak = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2
        # Calculate coincidences in the specified coincidence window
        window_low = config.low_coinc_windows
        window_high = config.high_coinc_windows
        coinc_window_mask = (bin_edges[:-1] >= window_low) & (bin_edges[:-1] < window_high)
        coincidences_rate = np.sum(histo_vals[coinc_window_mask]) / integration_time_s

        # Calculate rates for timestamps
        channel1_rate = len(timestamps1) / integration_time_s
        channel2_rate = len(timestamps2) / integration_time_s

        # Return results matching update_data naming
        return {
            "histo_vals": histo_vals,
            "bin_edges": bin_edges,
            "window_low": window_low,
            "window_high": window_high,
            "coincidences_rate": coincidences_rate,
            "channel1_rate": channel1_rate,
            "channel2_rate": channel2_rate,
        }
    

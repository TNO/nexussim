import time
import psutil
import pandas as pd
from datetime import datetime


def monitor_cpu_usage(
    process_name: str,
    sample_frequency: float,
    duration_minutes: float,
    output_file: str,
):
    """
    Monitor the CPU usage of a specific process at a given frequency for a certain duration.

    Args:
        process_name (str): Name of the program to monitor (e.g., 'firefox').
        sample_frequency (float): Sampling frequency in seconds.
        duration_minutes (float): Duration of monitoring in minutes.
        output_file (str): Path to save the resulting CSV file.
    """
    end_time = time.time() + (duration_minutes * 60)
    samples = []

    try:
        # Find the target process
        while time.time() < end_time:
            # Get all processes matching the process name
            processes = [
                p
                for p in psutil.process_iter(["name"])
                if process_name.lower() in p.info["name"].lower()
            ]

            if processes:
                cpu_usage = sum(
                    p.cpu_percent(interval=1) for p in processes
                )  # Sum CPU usage of all matching processes
            else:
                cpu_usage = 0.0  # If no such process exists, CPU usage is 0

            # Capture the timestamp and CPU usage
            timestamp = datetime.now().isoformat()
            print((timestamp, cpu_usage))
            samples.append({"timestamp": timestamp, "cpu_usage": cpu_usage})

            time.sleep(sample_frequency)

        # Save samples to a CSV file
        df = pd.DataFrame(samples)
        df.to_csv(output_file, index=False)
        print(f"CPU usage data saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Settings
process_name = "firefox"  # Process to monitor
sample_frequency = 1.0  # Frequency in seconds
duration_minutes = 1.0  # Duration in minutes
output_file = "cpu_usage_timeseries.csv"  # Output CSV file

# Start monitoring
monitor_cpu_usage(process_name, sample_frequency, duration_minutes, output_file)

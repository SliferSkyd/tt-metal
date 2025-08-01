import matplotlib

matplotlib.use("WebAgg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Path to the CSV file

df_without = pd.read_csv(
    "/localdev/sstanisic/tt-metal/generated/profiler/reports/2025_07_31_12_45_28/ops_perf_results_2025_07_31_12_45_28.csv"
)
df_clock = pd.read_csv(
    "/localdev/sstanisic/tt-metal/generated/profiler/reports/2025_07_31_11_52_40/ops_perf_results_2025_07_31_11_52_40.csv"
)
df_counter = pd.read_csv(
    "/home/sstanisic/local/tt-metal/generated/profiler/reports/2025_08_01_10_14_22/ops_perf_results_2025_08_01_10_14_22.csv"
)

# Select relevant columns (assuming column names)
columns_of_interest = [
    "OP CODE",
    "DEVICE FW DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
]

# df_without[["DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]"]] = df_without[["DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]"]] / 160
# df_clock[["DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]"]] = df_clock[["DEVICE COMPUTE CB WAIT FRONT [ns]", "DEVICE COMPUTE CB RESERVE BACK [ns]"]] / 160

df_comb = pd.concat(
    [
        df_without[["DEVICE FW DURATION [ns]"]],
        df_clock[["DEVICE COMPUTE CB WAIT FRONT [ns]"]],
        df_counter[["DEVICE COMPUTE CB WAIT FRONT [ns]"]],
    ],
    axis=1,
)

df_comb["RATIO"] = (
    df_clock["DEVICE COMPUTE CB WAIT FRONT [ns]"].values / df_counter["DEVICE COMPUTE CB WAIT FRONT [ns]"].values
)

print(df_comb.head(10).to_string(index=False))

ratio_avg = df_comb["RATIO"].mean()
ratio_var = df_comb["RATIO"].var()
print(f"\nRATIO avg: {ratio_avg:.6f}")
print(f"RATIO var: {ratio_var:.6f}")

pcc = df_clock["DEVICE COMPUTE CB WAIT FRONT [ns]"].corr(df_counter["DEVICE COMPUTE CB WAIT FRONT [ns]"])
print(f"\nPearson correlation coefficient (PCC): {pcc:.6f}")


x = df_counter["DEVICE COMPUTE CB WAIT FRONT [ns]"].values
y = df_clock["DEVICE COMPUTE CB WAIT FRONT [ns]"].values

mask = ~np.isnan(x) & ~np.isnan(y)
x_filtered = x[mask]
y_filtered = y[mask]

# Fit linear function: y = a*x + b
a, b = np.polyfit(x_filtered, y_filtered, 1)
print(f"\nLinear fit: f(x) = {a:.6f} * x + {b:.6f}")

# Optional: check fit quality
y_pred = a * x_filtered + b
mse = np.mean((y_filtered - y_pred) ** 2)
print(f"Mean squared error of fit: {mse:.6f}")
# print(df_with[columns_of_interest].to_string(index=False))

# Show filtered counter, clock, and predicted values in columns
df_compare = pd.DataFrame({"COUNTER_FILTERED": x_filtered, "CLOCK_FILTERED": y_filtered, "PREDICTED": y_pred})
print("\nFiltered and predicted values:")
print(df_compare.head(10).to_string(index=False))

plt.figure(figsize=(12, 6))
plt.plot(x_filtered, label="COUNTER_FILTERED", color="blue")
plt.plot(y_filtered, label="CLOCK_FILTERED", color="orange")
plt.plot(y_pred, label="PREDICTED", color="green", linestyle="--")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Timeseries of COUNTER_FILTERED, CLOCK_FILTERED, and PREDICTED")
plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files
cas1 = pd.read_csv(
    "CAS_1.csv",
    header=None,
    names=["Time", "CAS1"]
)

isis = pd.read_csv(
    "CAS_ISIS.csv",
    header=None,
    names=["Time", "CAS_ISIS"]
)

# Sort both datasets by time
cas1 = cas1.sort_values("Time")
isis = isis.sort_values("Time")

# Plot both curves
plt.figure(figsize=(12, 6))

plt.plot(cas1["Time"], cas1["CAS1"],
         label="CAS 1", color="blue")

plt.plot(isis["Time"], isis["CAS_ISIS"],
         label="CAS ISIS", color="red")

plt.xlabel("Time elapsed from 02:10:00 (seconds)")
plt.ylabel("Calibrated Airspeed (kt)")
plt.title("AF447: Comparison of Airspeed Measurements")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#________________________________________________________________________________

# Handle duplicate timestamps by averaging their values
cas1 = cas1.groupby("Time", as_index=False).mean()
isis = isis.groupby("Time", as_index=False).mean()

# Create a common time axis at 0.1-second intervals
start = max(cas1["Time"].min(), isis["Time"].min(), 0)
end = min(cas1["Time"].max(), isis["Time"].max())

time = np.arange(start, end, 0.1)

# Interpolate both datasets onto the common time axis
cas1_interp = np.interp(
    time, cas1["Time"], cas1["CAS1"]
)

isis_interp = np.interp(
    time, isis["Time"], isis["CAS_ISIS"]
)

# Calculate signed and absolute differences
delta_v = cas1_interp - isis_interp
D = np.abs(delta_v)

# Create comparison dataframe
comparison = pd.DataFrame({
    "Time": time,
    "CAS1": cas1_interp,
    "CAS_ISIS": isis_interp,
    "Delta_V": delta_v,
    "Absolute_Disagreement": D
})

comparison.to_csv("AF447_CAS_comparison.csv", index=False)

# Calculate statistics
print("Mean absolute disagreement:", np.mean(D), "kt")
print("Maximum disagreement:", np.max(D), "kt")
print("Median disagreement:", np.median(D), "kt")
print("RMSE:", np.sqrt(np.mean(delta_v**2)), "kt")

max_index = np.argmax(D)

print("Maximum disagreement occurs at:",
      time[max_index], "seconds")

print("CAS 1 at that time:", cas1_interp[max_index], "kt")
print("CAS ISIS at that time:", isis_interp[max_index], "kt")

# Plot signed difference
plt.figure(figsize=(12, 6))
plt.plot(time, delta_v, color="purple")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("Time elapsed from 02:10:00 (seconds)")
plt.ylabel("CAS 1 - CAS ISIS (kt)")
plt.title("Difference Between Airspeed Measurements")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/AF447_deltaV.png", dpi=180)
plt.show()

# Plot absolute disagreement
plt.figure(figsize=(12, 6))
plt.plot(time, D, color="darkorange")
plt.xlabel("Time elapsed from 02:10:00 (seconds)")
plt.ylabel("Absolute disagreement (kt)")
plt.title("Absolute Disagreement Between Airspeed Measurements")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/AF447_absolute_disagreement.png", dpi=180)
plt.show()




#_______________________________________________________________________________

# Load combined comparison data
df = pd.read_csv("AF447_CAS_comparison.csv")

# Plot absolute disagreement
plt.figure(figsize=(13, 6))

plt.plot(
    df["Time"],
    df["Absolute_Disagreement"],
    color="darkorange",
    linewidth=1.5,
    label="Absolute disagreement"
)

# Mark the 20 kt analysis threshold
plt.axhline(
    20,
    color="red",
    linestyle="--",
    label="20 kt analysis threshold"
)

# Mark maximum disagreement
max_idx = df["Absolute_Disagreement"].idxmax()

max_time = df.loc[max_idx, "Time"]
max_value = df.loc[max_idx, "Absolute_Disagreement"]

plt.scatter(max_time, max_value, color="black", zorder=5)

plt.annotate(
    f"Maximum: {max_value:.1f} kt",
    (max_time, max_value),
    xytext=(15, -25),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->")
)

plt.xlabel("Time elapsed from 02:10:00 (seconds)")
plt.ylabel("Absolute disagreement (kt)")
plt.title("AF447: Absolute Disagreement Between Airspeed Indications")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("plots/AF447_disagreement_analysis.png", dpi=200)
plt.show()

#________________________________________________________________________________

plt.figure(figsize=(13, 6))

plt.plot(
    comparison["Time"],
    comparison["Delta_V"],
    color="purple",
    linewidth=1.5
)

plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(20, color="red", linestyle="--", alpha=0.7)
plt.axhline(-20, color="red", linestyle="--", alpha=0.7)

plt.xlabel("Time elapsed from 02:10:00 (seconds)")
plt.ylabel("CAS 1 - CAS ISIS (kt)")
plt.title("Signed Difference Between Airspeed Indications")

plt.grid(True)
plt.tight_layout()
plt.savefig("plots/AF447_signed_difference.png", dpi=200)
plt.show()


#________________________________________________________________________________
# Identify periods where absolute disagreement exceeds 20 kt

threshold = 20

df["Significant_Disagreement"] = df["Absolute_Disagreement"] > threshold

# Total percentage of sampled points above threshold
percentage = df["Significant_Disagreement"].mean() * 100

print(f"\nPercentage of samples with disagreement > 20 kt: {percentage:.2f}%")

# Find continuous intervals above threshold
df["Group"] = (
    df["Significant_Disagreement"] !=
    df["Significant_Disagreement"].shift()
).cumsum()

intervals = df[df["Significant_Disagreement"]].groupby("Group")

print("\nIntervals where disagreement exceeds 20 kt:")

for _, group in intervals:
    start = group["Time"].iloc[0]
    end = group["Time"].iloc[-1]

    print(f"{start:.2f} s to {end:.2f} s")




#____________________________________________________________________________________
print("\nDISAGREEMENT BY INTERVAL")

interval_ranges = [
    (6.9, 8.0),
    (8.3, 15.7),
    (34.6, 66.5)
]

for start, end in interval_ranges:

    segment = df[
        (df["Time"] >= start) &
        (df["Time"] <= end)
    ]

    print(f"\nInterval: {start}–{end} seconds")
    print("Mean disagreement:",
          segment["Absolute_Disagreement"].mean(), "kt")

    print("Maximum disagreement:",
          segment["Absolute_Disagreement"].max(), "kt")

    print("Minimum disagreement:",
          segment["Absolute_Disagreement"].min(), "kt")
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "credit_predictions_log.csv",
    on_bad_lines="skip"   # skip malformed rows
)


plt.hist(df["default_probability"], bins=20, density=True)
plt.title("Default probability distribution")
plt.xlabel("Predicted probability")
plt.ylabel("Density")
plt.savefig("default_probability_distribution.png", dpi=150, bbox_inches="tight")
print("Saved default_probability_distribution.png")

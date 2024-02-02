import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from gdrn_simple.Metrics import ErrorRecordHeader as ERH 
import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (9, 5),
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
}
pylab.rcParams.update(params)

adjust_params = dict(top=0.93, bottom=0.088, left=0.125, right=0.9, hspace=0.4, wspace=0.25)


# error_path = "./output/gdrn/ambf_suturing/classAware_ambf_suturing_v0.0.1/inference_model_final/ambf_suturing_test/error_metrics.csv" 
error_path = "./output/gdrn/ambf_suturing/classAware_ambf_suturing_v0.0.2/inference_model_final_v002_with_det/ambf_suturing_test/error_metrics.csv" 

def main(error_path):
    error_path = Path(error_path)
    assert error_path.exists(), f"Error path {error_path} does not exist"

    df = pd.read_csv(error_path)


    # fig, ax = plt.subplots(1,3)
    # fig.suptitle(f"GDRN error metrics N={df.shape[0]}")
    # fig.set_tight_layout(True)
    # fig.subplots_adjust(**adjust_params)
    # sns.boxplot(df, y="re", ax=ax[0])
    # ax[0].set_ylabel(ERH.re.value)
    # sns.boxplot(df, y="te", ax=ax[1])
    # ax[1].set_ylabel(ERH.te.value)
    # sns.boxplot(df, y="mssd", ax=ax[2])
    # ax[2].set_ylabel(ERH.mssd.name+" (mm)")

    # [a.grid() for a in ax]

    # sns.violinplot(df, y="re", ax=ax[1])

    # sns.swarmplot(df, y="re", ax=ax)
    # sns.swarmplot(df, x="guidance", y="total_errors", color="black", ax=ax, order=["Baseline","Visual","Haptic","Audio"])

    re_median = df["re"].median()
    te_median = df["te"].median()
    mssd_median = df["mssd"].median()

    fig, ax = plt.subplots(3,1)
    fig.suptitle(f"GDRN error metrics N={df.shape[0]}")
    # fig.set_tight_layout(True)
    fig.subplots_adjust(**adjust_params)
    sns.histplot(df, x="re", ax=ax[0], kde=True, bins=100)
    ax[0].set_xlabel(ERH.re.value)
    ax[0].axvline(re_median, color="red", label="median", linestyle="dashed")
    sns.histplot(df, x="te", ax=ax[1], kde=True, bins=100)
    ax[1].set_xlabel(ERH.te.value)
    ax[1].axvline(te_median, color="red", label="median", linestyle="dashed")
    sns.histplot(df, x="mssd", ax=ax[2], kde=True, bins=100 )
    ax[2].set_xlabel(ERH.mssd.name+" (mm)")
    ax[2].axvline(mssd_median, color="red", label="median", linestyle="dashed")

    [a.grid() for a in ax]
    plt.show()

if __name__ == "__main__":
    main(error_path)


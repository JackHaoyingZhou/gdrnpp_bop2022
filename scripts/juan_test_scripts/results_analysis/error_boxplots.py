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

adjust_params = dict(top=0.88, bottom=0.18, left=0.125, right=0.9, hspace=0.2, wspace=0.25)


error_path = "./output/gdrn/ambf_suturing/classAware_ambf_suturing_v0.0.1/inference_model_final/ambf_suturing_test/error_metrics.csv" 

def main(error_path):
    error_path = Path(error_path)
    assert error_path.exists(), f"Error path {error_path} does not exist"

    df = pd.read_csv(error_path)


    fig, ax = plt.subplots(1,3)
    fig.suptitle(f"GDRN error metrics N={df.shape[0]}")
    fig.set_tight_layout(True)
    fig.subplots_adjust(**adjust_params)
    sns.boxplot(df, y="re", ax=ax[0])
    ax[0].set_ylabel(ERH.re.value)
    sns.boxplot(df, y="te", ax=ax[1])
    ax[1].set_ylabel(ERH.te.value)
    sns.boxplot(df, y="mssd", ax=ax[2])
    ax[2].set_ylabel(ERH.mssd.name+" (mm)")

    [a.grid() for a in ax]

    # sns.violinplot(df, y="re", ax=ax[1])

    # sns.swarmplot(df, y="re", ax=ax)
    # sns.swarmplot(df, x="guidance", y="total_errors", color="black", ax=ax, order=["Baseline","Visual","Haptic","Audio"])

    plt.show()

if __name__ == "__main__":
    main(error_path)


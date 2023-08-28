from collections import defaultdict
import click
import mmcv
import json
import matplotlib.pyplot as plt

# default_path = "./output/gdrn/tudl/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl_juan/metrics.json"
default_path = "./output/gdrn/ambf_suturing/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ambf_suturing/metrics.json"
# default_path = "./output/gdrn/ambf_suturing/classAware_ambf_suturing_env1_automated1/metrics.json"

def on_close(event):
    size = event.canvas.figure.get_size_inches()  # Get the size in inches
    print("Resized Figure Size:", size)
    plt.ioff()  # Turn off interactive mode to ensure the script waits for the plot to be closed

@click.command()
@click.argument("metrics_path", type=str, default=default_path)
def visualize_metrics(metrics_path:str):
    # metrics = mmcv.load(metrics_path)

    my_metrics = ["loss_PM_R", "loss_centroid", "loss_coor_x", "loss_coor_y", "loss_coor_z", "loss_mask"]
    metrics_parsed = defaultdict(list) 
    with open(metrics_path, "r") as f:
        for line in f:
            metrics = json.loads(line)
            metrics_parsed["iteration"].append(metrics["iteration"])
            for m in my_metrics:
                metrics_parsed[m].append(metrics[m][0])

            # break
            x = 0

    fig, ax = plt.subplots(3,2)
    fig.canvas.mpl_connect('close_event', on_close)
    fig.set_size_inches(12.57,5.92)
    fig.subplots_adjust(hspace=0.38)

    for m1, a1 in zip(my_metrics, ax.flatten()):
        a1.plot(metrics_parsed["iteration"], metrics_parsed[m1])
        a1.set_title(m1)
        a1.grid(True)
    plt.show()

if __name__ =="__main__":
    visualize_metrics()

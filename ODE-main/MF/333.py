import matplotlib.pyplot as plt


def plot_bar_chart(data, x_labels, y_label, title, colors, y_start=None, x_tick_rotation=0, x_tick_fontsize=10, bar_width=0.6):
    """
    Plots a bar chart with given data and labels.

    Parameters:
    - data: List of values for the bars.
    - x_labels: List of labels for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the bar chart.
    - x_tick_rotation: Rotation angle for x-axis labels (optional).
    - x_tick_fontsize: Font size for x-axis labels (optional).
    """

    if len(data) != len(x_labels):
        raise ValueError("Length of data and x_labels should be the same.")

    plt.bar(x_labels, data, width=bar_width)
    plt.bar(x_labels, data, color=colors)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(x_labels, rotation=x_tick_rotation, fontsize=x_tick_fontsize)
    if y_start is not None:
        plt.ylim(bottom=y_start)
    plt.savefig('different samples.png')
    plt.show()


# Sample data and labels
data = [0.3635, 0.3003, 0.2533, 0.2008, 0.1785, 0.1553, 0.1636, 0.1987, 0.1933, 0.2134]
x_labels = ["1e^-5%", "1e-4", "1e-3%", "1e-2%", "1e-1%", "1%", "2%", "3%", "4%", "5%"]
y_label = "HR@50"
colors = ["pink", "pink", "pink", "pink", "pink", "pink", "pink", "pink", "pink", "pink"]
title = ""

plot_bar_chart(data, x_labels, y_label, title, colors, y_start=0.1, x_tick_fontsize=7, bar_width=0.6)

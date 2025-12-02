import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from brokenaxes import (brokenaxes)

class AtmosphericStratification:
    def __init__(self):
        # Define temperature profile and altitude
        self.temperature = [-60, -56, -56, -2, -2, -80, -80, -65, -55, -40, -25]
        self.altitude = [0, 11, 20, 47, 51, 80, 85, 135, 350, 400, 450]

        # Define layer boundaries
        self.layers = {
            "Troposphere": (0, 11),
            "Stratosphere": (11, 50),
            "Mesosphere": (50, 85),
            "Thermosphere": (85, 400),
            "Exosphere": (400, 450)
        }

        self.layer_colors = {
            "Troposphere": "#ffc345",
            "Stratosphere": "#ff9e4a",
            "Mesosphere": "#ff785e",
            "Thermosphere": "#614e72",
            "Exosphere": "#214675"
        }


        # Labeling
        self.layer_labels = {
            "Troposphere": 5,
            "Stratosphere": 30,
            "Mesosphere": 67,
            "Thermosphere": 380,
            "Exosphere": 420
        }

    def plot_layers(self):

        bax = brokenaxes(ylims=((0, 120), (370, 450)), hspace=.05)

        # Add filled colored regions for atmospheric layers
        for layer, (bottom, top) in self.layers.items():
            for ax in bax.axs:
                y0, y1 = ax.get_ylim()
                if top < y0 or bottom > y1:
                    continue  # not in this axis range
                draw_bottom = max(bottom, y0)
                draw_top = min(top, y1)
                rect = Rectangle((-100, draw_bottom), 120, draw_top - draw_bottom,
                                 color=self.layer_colors[layer], alpha=0.4)
                ax.add_patch(rect)

        # Create gradient line manually using plot()
        temp = np.array(self.temperature)
        alt = np.array(self.altitude)
        for i in range(len(temp) - 2):  # skip last segment (exo)
            t1, t2 = temp[i], temp[i + 1]
            a1, a2 = alt[i], alt[i + 1]

            # Determine which axis this segment belongs to
            for ax in bax.axs:
                y0, y1 = ax.get_ylim()
                if y0 <= a1 <= y1 or y0 <= a2 <= y1:
                    fade = i / (len(temp) - 2)
                    color = (0.1 + 0.6 * fade, 0.3 + 0.6 * fade, 0.6 + 0.4 * fade)
                    ax.plot([t1, t2], [a1, a2], color=color, linewidth=3)
                    break

        # Dashed exosphere line
        t1, t2 = self.temperature[-2], self.temperature[-1]
        a1, a2 = self.altitude[-2], self.altitude[-1]
        for ax in bax.axs:
            y0, y1 = ax.get_ylim()
            if y0 <= a1 <= y1 or y0 <= a2 <= y1:
                ax.plot([t1, t2], [a1, a2], linestyle='--', color='white', linewidth=2.5, zorder=100)
                break

        # Add layer boundary lines
        for boundary in [11, 50, 85]:
            for ax in bax.axs:
                y0, y1 = ax.get_ylim()
                if y0 < boundary < y1:  # Avoid edge duplication
                    ax.axhline(y=boundary, color='k', linestyle='--', linewidth=1)
                    break

        for layer, label_alt in self.layer_labels.items():
            for ax in bax.axs:
                y0, y1 = ax.get_ylim()
                if y0 <= label_alt <= y1:
                    ax.text(-95, label_alt, layer, fontsize=10, va='center', weight='bold')
                    break

        # Axis labels and formatting
        bax.set_xlabel("Temperature (°C)")
        bax.set_ylabel("Altitude (km)")
        bax.set_xlim(-100, 20)
        #bax.set_ylim(0, 450)
        bax.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f'C:\\Users\\ninam\\Documents\\Chile Stuff\\Dissertation\\atmospheric_stratification.png')
        plt.show()

if __name__ == "__main__":
    lyr = AtmosphericStratification()
    lyr.plot_layers()

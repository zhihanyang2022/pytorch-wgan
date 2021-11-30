import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
import torch
import seaborn as sns


class Figure:

    def __init__(self, save_animation=False, save_name="animation"):

        self.figure, (self.ax1, self.ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
        sns.despine(self.figure)
        plt.subplots_adjust(bottom=0.15)
        plt.show(block=False)

        self.save_animation = save_animation

        if self.save_animation:
            self.writer = PillowWriter(fps=5)
            self.writer.setup(self.figure, f'{save_name}.gif', dpi=100)

    def replot(self, ws, real_data_all, fake_data_batch, critic):

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(ws)
        self.ax1.axhline(0, linestyle='--', color='red')

        self.ax1.set_xlabel("Steps")
        self.ax1.set_ylabel("W-Distance (Estimated)")

        self.ax1.set_ylim(-0.3, 3.5)

        xs = torch.from_numpy(np.linspace(0, 6, 100).reshape(-1, 1)).float()
        critic_values = critic(xs).detach().numpy().reshape(-1, )
        critic_values = (critic_values - np.min(critic_values)) / (np.max(critic_values) - np.min(critic_values))

        self.ax2.plot(xs, critic_values, label='Critic (Normalized)')
        sns.kdeplot(real_data_all.reshape(-1, ), ax=self.ax2, shade=True, label="Real KDE (All)", color='green')
        sns.kdeplot(fake_data_batch.detach().numpy().reshape(-1, ), ax=self.ax2, shade=True, label="Fake KDE (Batch)", color='red')

        plt.legend(loc='upper right')
        self.ax2.set_xlabel("Data")

        self.ax2.set_ylim(0, 1.5)
        self.ax2.set_xlim(0, 6)

        # It seems like draw() is not required, but I'm not going down the rabbit hole
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        if self.save_animation:
            self.writer.grab_frame()

    def finalize(self):
        if self.save_animation:
            self.writer.finish()

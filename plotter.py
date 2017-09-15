import numpy as np
from visdom import Visdom

class Plotter:
    def __init__(self, env, window_config_list, port=11119):
        self.vis = Visdom(port=port)
        self.env = env
        self.windows = self.register_windows(window_config_list)

    def register_windows(self, window_config_list):
        windows = {}
        for window_config in window_config_list:
            windows[window_config['opts']['title']] = {
                'plot': self.vis.line(
                    X=np.zeros(1),
                    Y=np.zeros(1),
                    opts=window_config['opts'],
                    env=self.env
                ),
                'update_mode': 'update'
            }

        return windows

    def update(self, stats_list):
        for stats in stats_list:
            window = self.windows[stats['title']]
            self.vis.line(
                X=np.array([stats['X']]),
                Y=np.array([stats['Y']]),
                win=window['plot'],
                env=self.env,
                update=window['update_mode']
            )
            window['update_mode'] = 'append'

    def save(self):
        self.vis.save([self.env])

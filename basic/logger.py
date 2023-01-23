import time
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir, env_name, algorithm):
        self.t0 = time.time()
        self.log_dir = log_dir
        # Make direectory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # Create log file
        self.log_file = os.path.join(self.log_dir, f'{env_name}_{algorithm}.log')
        with open(self.log_file, 'w') as f:
            f.write(f'epoch step time ep_len return\n')
        
    def log(self, epoch, step, avg_return, ep_len):
        t = time.time() - self.t0
        print(f'[{datetime.now()}] epoch: {epoch:<3} \t step: {step:<10} \t ep_len: {ep_len:.3f} \t return: {avg_return:.3f}')
        with open(self.log_file, 'a') as f:
            f.write(f'{epoch} {step} {t:.3f} {ep_len:.3f} {avg_return:.3f}\n')


def plot_logs(log_dir, env_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    # Load logs
    log_files = [f for f in os.listdir(log_dir) if f.startswith(env_name)]
    log_files.sort()
    print(log_files)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for log_file in log_files:
        df = pd.read_csv(os.path.join(log_dir, log_file), sep=' ')
        #df['return'] = df['return'].rolling(10).mean()
        #df['ep_len'] = df['ep_len'].rolling(10).mean()
        df.plot(x='step', y='return', ax=ax, label=log_file)
        #df.plot(x='step', y='ep_len', ax=ax[1], label=log_file)
    ax.set_title('Return')
    #ax[1].set_title('Episode Length')
    plt.show()

if __name__ == '__main__':
    from arguments import get_args
    args = get_args()
    plot_logs(args.log_dir, args.env_name)



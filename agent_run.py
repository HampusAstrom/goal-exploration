import wandb
import run_wandb_with_sweep_conf

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    parser.add_argument('sweep_id')
    args = parser.parse_args()

    wandb.agent(sweep_id=args.sweep_id,
                project=args.project,
                function=run_wandb_with_sweep_conf.main,
                )
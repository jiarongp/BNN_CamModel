import os
from train import train_eval
from experiment import experiment
from utils.misc import get_args, get_params

def main():
    try:
        args = get_args()
        print(args.params)
        params = get_params(args.params)
    except Exception as err:
        print(err)
        print("... missing or invalid arguments")
        exit(0)
    # create all dirs that are needed
    dirs = [params.log.log_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    # concatenate brands and models
    for b, m in zip(params.dataloader.brands, 
                    params.dataloader.models):
        params.dataloader.brand_models.append("_".join([b, m]))
    if params.run.train or params.run.evaluate:
        dirs = [params.trainer.ckpt_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        train_eval(params)

    if params.run.experiment:
        experiment(params)

if __name__ == '__main__':
    main()
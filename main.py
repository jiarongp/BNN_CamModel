import os
from train import train
from evaluate import evaluate
from experiment import stats
from utils.parser import get_args, get_params
import utils.telegram_bot as bot 

def main():
    debug = False
    try:
        if debug:
            params = get_params("params/ensemble_cnn.json")
        else:
            args = get_args()
            print(args.params)
            params = get_params(args.params)
    except Exception as err:
        print(err)
        print("... missing or invalid arguments")
        exit(0)

    dirs = [params.trainer.ckpt_dir, params.log.log_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    if params.run.train:
        # bot.sendtext("Start training {}".format(str(params.run.name)))
        train(params)
        bot.sendtext("Finished training {}".format(str(params.run.name)))

    if params.run.evaluate:
        evaluate(params)

    if params.run.stats:
        stats(params)

if __name__ == '__main__':
    main()

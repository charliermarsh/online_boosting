import argparse
from random import seed
from yaml import dump
from utils.experiment import test
from utils.utils import *


if __name__ == "__main__":

    dataset = "breast-cancer_scale.txt"

    parser = argparse.ArgumentParser(
        description='Test error for a combination of ensembler and weak learner.')
    parser.add_argument('ensembler', help='chosen ensembler')
    parser.add_argument('M', metavar='# weak_learners',
                        help='number of weak learners', type=int)
    parser.add_argument(
        'trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
    parser.add_argument('--record', action='store_const', const=True,
                        default=False, help='export the results in YAML format')
    args = parser.parse_args()

    ensembler = get_ensembler(args.ensembler)
    performance = {}
    performance_baseline = {}
    for weak_learner in weak_learners:
        data = load_data("data/" + dataset)
        seed(0)

        accuracy, baseline = test(
            ensembler, weak_learners[weak_learner], data, args.M, trials=args.trials)
        performance[weak_learner] = (accuracy[-1], baseline[-1])

    if args.record:
        results = performance
        results['m'] = args.M
        results['booster'] = args.ensembler
        filename = args.ensembler + "_ALL_" + str(args.M) + ".yml"
        f = open(filename, 'w+')
        f.write(dump(results))

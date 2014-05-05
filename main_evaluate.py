import argparse
from yaml import dump
from utils.experiment import test
from utils.utils import *


if __name__ == "__main__":
    dataset = "breast-cancer_scale.txt"

    parser = argparse.ArgumentParser(
        description='Test error for a combination of ensembler and weak learner.')
    parser.add_argument('ensembler', help='chosen ensembler')
    parser.add_argument('weak_learner', help='chosen weak learner')
    parser.add_argument('M', metavar='# weak_learners',
                        help='number of weak learners', type=int)
    parser.add_argument(
        'trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
    parser.add_argument('--record', action='store_const', const=True,
                        default=False, help='export the results in YAML format')
    args = parser.parse_args()

    ensembler = get_ensembler(args.ensembler)
    weak_learner = get_weak_learner(args.weak_learner)
    data = load_data("data/" + dataset)

    accuracy, baseline = test(
        ensembler, weak_learner, data, args.M, trials=args.trials)

    print accuracy
    print baseline[-1]

    if args.record:
        results = {
            'm': args.M,
            'accuracy': accuracy,
            'baseline': baseline[-1],
            'booster': args.ensembler,
            'weak_learner': args.weak_learner,
        }
        filename = args.ensembler + "_" + \
            args.weak_learner + "_" + str(args.M) + ".yml"
        f = open(filename, 'w+')
        f.write(dump(results))

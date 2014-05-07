import argparse
from random import seed
from yaml import dump
from utils.experiment import testNumLearners
from utils.utils import *

if __name__ == "__main__":
    seed(0)

    parser = argparse.ArgumentParser(
        description='Test error for a combination of ensembler and weak learner.')
    parser.add_argument('dataset', help='dataset filename')
    parser.add_argument('ensembler', help='chosen ensembler')
    parser.add_argument('weak_learner', help='chosen weak learner')
    parser.add_argument(
        'start', help='initial number of weak learners', type=int)
    parser.add_argument('end', help='final number of weak learners', type=int)
    parser.add_argument(
        'inc', help='increment for number of weak learners', type=int)
    parser.add_argument('--record', action='store_const',
                        const=True, default=False, help='export the results in YAML format')
    parser.add_argument(
        'trials', help='number of trials (each with different shuffling of the data); defaults to 1', type=int, default=1, nargs='?')
    args = parser.parse_args()

    ensembler = get_ensembler(args.ensembler)
    weak_learner = get_weak_learner(args.weak_learner)
    data = load_data("data/" + args.dataset)

    accuracy = testNumLearners(
        ensembler, weak_learner, data, args.start, args.end, args.inc, trials=args.trials)

    print accuracy

    if args.record:
        results = {
            'accuracy': accuracy,
            'booster': args.ensembler,
            'weak_learner': args.weak_learner,
            'trials': args.trials,
            'seed': 0
        }
        filename = args.ensembler + "_" + \
            args.weak_learner + "_" + \
            str(args.start) + "_" + str(args.end) + \
            "_" + str(args.inc) + ".yml"
        f = open(filename, 'w+')
        f.write(dump(results))

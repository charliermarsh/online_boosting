import argparse
from random import seed
from yaml import dump
from utils.experiment import test
from utils.utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test error for every combination of ensembler and weak learner.')
    parser.add_argument('dataset', help='dataset filename')
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
        data = load_data(args.dataset)
        seed(0)

        try:
            accuracy, baseline = test(
                ensembler, weak_learners[weak_learner], data, args.M, trials=args.trials)
            performance[weak_learner] = (accuracy[-1], baseline[-1])
        except AttributeError:
            pass

    print "Accuracy:"
    print performance

    if args.record:
        results = performance
        results['m'] = args.M
        results['booster'] = args.ensembler
        results['dataset'] = args.dataset
        results['trials'] = args.trials
        results['seed'] = 0
        dataset_abbrev = args.dataset.split('/')[-1].split('.')[-2]
        filename = args.ensembler + "_ALL_" + \
            str(args.M) + "_" + dataset_abbrev + ".yml"
        f = open(filename, 'w+')
        f.write(dump(results))

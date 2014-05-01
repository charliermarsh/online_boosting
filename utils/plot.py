import sys
import pylab as pl
from yaml import load

if __name__ == "__main__":
    colors = ['r', 'g', 'b', 'y']

    pl.ylim([0.5, 1])
    pl.xlabel('Number of Examples')
    pl.ylabel('Error Rate')
    pl.title('Error Rate Over Time')

    i = 0
    for filename in sys.argv[1:]:
        data = load(open(filename).read())
        ys = data['accuracy']
        xs = range(len(ys))
        pl.plot(xs, ys, colors[i] + '--')
        i = (i + 1) % len(colors)
    pl.show()

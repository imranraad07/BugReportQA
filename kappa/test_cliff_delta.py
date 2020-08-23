from __future__ import division


def cliffsDelta(lst1, lst2, **dull):
    """Returns delta and true if there are more than 'dull' differences"""
    if not dull:
        dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474}  # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j * repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j) * repeats
    d = (more - less) / (m * n)
    size = lookup_size(d, dull)
    return d, size


def lookup_size(delta: float, dull: dict) -> str:
    """
    :type delta: float
    :type dull: dict, a dictionary of small, medium, large thresholds.
    """
    delta = abs(delta)
    if delta < dull['small']:
        return 'negligible'
    if dull['small'] <= delta < dull['medium']:
        return 'small'
    if dull['medium'] <= delta < dull['large']:
        return 'medium'
    if delta >= dull['large']:
        return 'large'


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


def test_small_difference():
    """ test data from Marco Torchiano R package https://github.com/mtorchiano/effsize"""
    treatment = [10, 10, 20, 20, 20, 30, 30, 30, 40, 50]
    control = [10, 20, 30, 40, 40, 50]

    d, res = cliffsDelta(treatment, control)
    # Cliff's Delta
    #
    # delta estimate: -0.25 (small)
    # 95 percent confidence interval:
    #  inf        sup
    # -0.7265846  0.3890062
    # self.assertEqual("small", res)
    # self.assertEqual(d, -0.25)
    print(res, d)


def test_tim():
    lst1 = range(8)
    out = []
    expected = ['negligible', 'negligible', 'small', 'small', 'medium']
    for r in [1.01, 1.1, 1.21, 1.5, 2]:
        lst2 = list(map(lambda x: x * r, lst1))
        d, res = cliffsDelta(lst1, lst2)
        out.append(res)

    # self.assertEqual(expected, out)
    print(expected, out)


def test_negligible():  # Marco
    x1 = [10, 20, 20, 20, 30, 30, 30, 40, 50, 100]
    x2 = [10, 20, 30, 40, 40, 50]
    d, res = cliffsDelta(x1, x2)
    # self.assertAlmostEqual(-0.06667, d, 4)
    print(-0.06667, d, 4)


def test_nonoverlapping():  # marco's test
    x1 = [10, 20, 20, 20, 30, 30, 30, 40, 50, 100]
    x2 = [10, 20, 30, 40, 40, 50]
    factor = 110
    x2 = [x + factor for x in x2]

    d, res = cliffsDelta(x1, x2)
    # self.assertEqual(res, 'large')
    # self.assertAlmostEqual(d, -1, 2)
    print(res, 'large')
    print(d, -1, 2)


import pandas as pd

df_random = pd.read_csv("random.csv")
df_util = pd.read_csv("util_only.csv")
df_compat = pd.read_csv("compat_only.csv")
df_lucene = pd.read_csv("lucene.csv")
df_evpi = pd.read_csv("evpi.csv")

if __name__ == '__main__':
    effect_size = ['mrr', 'p_1', 'p_3', 'p_5']
    for dataset in effect_size:
        print(dataset)
        x1 = df_random[dataset]
        x2 = df_evpi[dataset]
        d, res = cliffsDelta(x2, x1)
        print('random', d.__round__(2), res)

        x1 = df_lucene[dataset]
        x2 = df_evpi[dataset]
        d, res = cliffsDelta(x2, x1)
        print('lucene', d.__round__(2), res)

        x1 = df_util[dataset]
        x2 = df_evpi[dataset]
        d, res = cliffsDelta(x2, x1)
        print('util', d.__round__(2), res)

        x1 = df_compat[dataset]
        x2 = df_evpi[dataset]
        d, res = cliffsDelta(x2, x1)
        print('compat', d.__round__(2), res)


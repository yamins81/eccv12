import genson
from eccv12.toyproblem import *

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

config_tiny=dict(
        n_examples=100,
        n_folds=2,
        feat_spec=dict( seed=1, n_features=5, scale=2.2),
        split_decisions=None,
        save_svms=True
        )

def test_random_train_test_idxs():
    for n_examples in 66, 100:
        train, test = random_train_test_idxs(n_examples, n_fold=4, split_idx=2)
        assert len(set(train)) == len(train)
        assert len(set(test)) == len(test)
        assert not set(train).intersection(test)

    # -- the test sets cover the whole range, and do not overlap
    tests = [random_train_test_idxs(50, 5, ii)[1] for ii in range(5)]
    tests = np.asarray(tests)
    assert tests.size == 50
    assert set(tests.flatten()) == set(range(50))

    # -- when the number of examples doesn't fit evenly into the folds,
    #    some examples are dropped.
    tests = [random_train_test_idxs(54, 5, ii)[1] for ii in range(5)]
    tests = np.asarray(tests)
    assert tests.size == 50
    assert len(set(tests.flatten())) == 50


def test_screening_prog():
    # smoke test
    prog = screening_prog(**config_tiny)
    print genson.dumps(prog)
    rval = JSONFunction(prog)()
    print rval
    assert 'loss' in rval
    assert 'decisions' in rval
    assert len(rval['splits']) == 2
    assert rval['splits'][0] != rval['splits'][1]


def test_replicable():
    ctrls = [Ctrl() for ii in range(3)]

    def foo(ctrl):
        BI = BoostableDigits()
        return BI.evaluate(config_tiny, ctrl)
    r0, r1, r2 = map(foo, ctrls)

    for f in [
            lambda r: r['splits'][0]['train_prediction'],
            lambda r: r['splits'][1]['train_prediction'],
            lambda r: r['splits'][0]['test_prediction'],
            lambda r: r['splits'][1]['test_prediction'],
            ] :
        assert f(r0) == f(r1)
        assert f(r0) == f(r2)

    svm0 = cPickle.loads(ctrls[0].attachments['svm_0'])
    svm1 = cPickle.loads(ctrls[1].attachments['svm_0'])
    assert np.allclose(svm0['weights'], svm1['weights'])

    assert r0 == r1


def test_decisions_do_something():
    config=dict(seed=1, n_features=5, scale=2.2)
    X, y = digits_xy()

    ctrl0 = Ctrl()
    ctrl1 = Ctrl()

    B0 = BoostableDigits()
    B0.split_decisions = None
    r0 = B0.evaluate(config, ctrl0)

    B1 = BoostableDigits()
    B1.split_decisions = np.random.RandomState(32).randn(5, len(y))
    r1 = B1.evaluate(config, ctrl1)

    # contrast test_replicable where they come out the same
    svm0 = cPickle.loads(ctrl0.attachments['svm'])
    svm1 = cPickle.loads(ctrl1.attachments['svm'])
    assert not np.allclose(svm0['weights'], svm1['weights'])


def test_boosting_margin_goes_down():
    X, y = digits_xy()
    n_rounds = 8
    n_features_per_round = 16
    split_decisions = None
    margins = []
    for round_ii in range(n_rounds):
        BI = BoostableDigits()
        BI.split_decisions = split_decisions
        result = BI.evaluate(
                config=dict(
                    seed=round_ii,
                    n_features=n_features_per_round,
                    scale=2.2),
                ctrl=Ctrl())
        split_decisions = np.asarray(result['decisions'])
        assert len(split_decisions) == 5
        assert split_decisions.ndim == 2
        print 'mean abs decisions', abs(split_decisions).mean(),
        margins.append(1 - np.minimum(split_decisions * y, 1).mean())
        for key in 'train_accuracy', 'test_accuracy':
            print key, np.mean([r[key] for r in result['split']]),
        print ''
    print margins
    print list(reversed(margins))
    print list(sorted(margins))
    assert list(reversed(margins)) == list(sorted(margins))


def test_boosting_for_smoke():
    X, y = digits_xy()
    print len(y)

    n_rounds = 16
    n_features_per_round = 16

    # -- train jointly
    print 'Training jointly'
    BI = BoostableDigits()
    BI.split_decisions = None
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=n_rounds * n_features_per_round,
                scale=2.2),
            ctrl=Ctrl())
    decisions = np.asarray(result['decisions'])
    print decisions.shape
    print 'mean abs decisions', abs(decisions).mean(),
    print 'mean margins', 1 - np.minimum(decisions * y, 1).mean(),
    tr_acc = np.mean([r['train_accuracy'] for r in result['split']])
    te_acc = np.mean([r['test_accuracy'] for r in result['split']])
    print 'train_accuracy', tr_acc,
    print 'test_accuracy', te_acc,
    print ''

    # -- train just one
    print 'Training one round'
    BI = BoostableDigits()
    BI.split_decisions = None
    result = BI.evaluate(
            config=dict(
                seed=1,
                n_features=n_features_per_round,
                scale=2.2),
            ctrl=Ctrl())
    decisions = np.asarray(result['decisions'])
    print 'mean abs decisions', abs(decisions).mean(),
    print 'mean margins', 1 - np.minimum(decisions * y, 1).mean(),
    one_tr_acc = np.mean([r['train_accuracy'] for r in result['split']])
    one_te_acc = np.mean([r['test_accuracy'] for r in result['split']])
    print 'train_accuracy', one_tr_acc,
    print 'test_accuracy', one_te_acc,
    print ''

    # -- train in rounds
    print 'Training in rounds'
    split_decisions = None
    for round_ii in range(n_rounds):
        BI = BoostableDigits()
        BI.split_decisions = split_decisions
        result = BI.evaluate(
                config=dict(
                    seed=round_ii,
                    n_features=n_features_per_round,
                    scale=2.2),
                ctrl=Ctrl())
        split_decisions = np.asarray(result['decisions'])
        assert len(split_decisions) == 5
        assert split_decisions.ndim == 2
        print 'mean abs decisions', abs(split_decisions).mean(),
        print 'mean margins', 1 - np.minimum(split_decisions * y, 1).mean(),
        round_tr_acc = np.mean([r['train_accuracy'] for r in result['split']])
        round_te_acc = np.mean([r['test_accuracy'] for r in result['split']])
        print 'train_accuracy', round_tr_acc,
        print 'test_accuracy', round_te_acc,
        print ''

    # assert that round-training and joint training are both way better than
    # training just one
    assert tr_acc > 90
    assert round_tr_acc > 90
    assert one_tr_acc < 70


def test_random_search_boosting():
    X, y = digits_xy()
    print X.shape
    print len(y)

    def foo(n_candidates):

        n_rounds = 16
        rstate = np.random.RandomState(123)  # for sampling configs

        # On every round of boosting,
        # draw some candidates, and select the best among them to join the
        # ensemble.

        selected = []
        for round_ii in range(n_rounds):
            print 'ROUND', round_ii
            candidates = []

            # -- Here is where we would use hyperopt
            #    For simplicity, we do a random search here.
            for candidate_ii in range(n_candidates):
                print ' CANDIDATE', candidate_ii
                BI = BoostableDigits()
                if selected:
                    BI.split_decisions = selected[-1]['decisions']
                else:
                    BI.split_decisions = None
                result = BI.evaluate(
                        config=dict(
                            seed=int(rstate.randint(2**31)),
                            n_features=int(rstate.randint(2, 32)),
                            scale=float(np.exp(rstate.randn())),
                            ),
                        ctrl=Ctrl())
                print '  loss', result['loss']
                candidates.append(result)

            # loss is the average test set err across the internal folds
            loss = np.array([rr['loss'] for rr in candidates])
            selected_ind = loss.argmin()
            best = candidates[selected_ind]
            if len(selected):
                # XXX: what are we supposed to do here?
                # among the pool of candidates it is not necessarily the case
                # that any of them makes an improvement on the validation set
                if best['loss'] < selected[-1]['loss']:
                    selected.append(best)
                else:
                    print 'SKIPPING CRAPPY BEST CANDIDATE'
                    pass
            else:
                selected.append(best)

            split_decisions = best['decisions']
            print 'mean margins', 1 - np.minimum(split_decisions * y, 1).mean(),
            round_tr_acc = np.mean([rr['train_accuracy'] for rr in best['split']])
            round_te_acc = np.mean([rr['test_accuracy'] for rr in best['split']])
            print 'train_accuracy', round_tr_acc,
            print 'test_accuracy', round_te_acc,
            print ''
        return selected

    r1 = foo(1)
    r2 = foo(2)
    r5 = foo(5)

    assert r1[-1]['loss'] > r2[-1]['loss']
    assert r2[-1]['loss'] > r5[-1]['loss']


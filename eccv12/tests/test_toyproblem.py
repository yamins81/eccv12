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
    rval = JSONFunction(prog)(ctrl=Ctrl())
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
    ctrl0 = Ctrl()
    ctrl1 = Ctrl()

    B0 = BoostableDigits()
    B0.split_decisions = None
    r0 = B0.evaluate(config_tiny, ctrl0)

    rstate = np.random.RandomState(32)
    B1 = BoostableDigits()
    ctrl1.attachments['split_decisions'] = rstate.randn(
            config_tiny['n_folds'],
            config_tiny['n_examples'])
    r1 = B1.evaluate(config_tiny, ctrl1)

    # contrast test_replicable where they come out the same
    svm0 = cPickle.loads(ctrl0.attachments['svm_0'])
    svm1 = cPickle.loads(ctrl1.attachments['svm_0'])
    assert not np.allclose(svm0['weights'], svm1['weights'])


def test_boosting_margin_goes_down():
    n_examples = 1750
    X, y = digits_xy(n_examples)
    n_rounds = 8
    margins = []
    ctrl = Ctrl()
    ctrl.attachments['split_decisions'] = None
    for round_ii in range(n_rounds):
        BI = BoostableDigits()
        config = dict(
            n_examples=n_examples,
            n_folds=5,
            feat_spec=dict(seed=round_ii, n_features=16, scale=2.2),
            split_decisions=None, # gets clobbered by ctrl attachment
            save_svms=False       # good to test False sometimes
            )
        result = BI.evaluate(config, ctrl)
        split_decisions = np.asarray(result['decisions'])
        ctrl.attachments['split_decisions'] = split_decisions
        assert len(split_decisions) == 5
        assert split_decisions.ndim == 2
        print 'mean abs decisions', abs(split_decisions).mean(),
        margins.append(1 - np.minimum(split_decisions * y, 1).mean())
        for key in 'train_accuracy', 'test_accuracy':
            print key, np.mean([rr[key] for rr in result['splits']]),
        print ''
    print margins
    print list(reversed(margins))
    print list(sorted(margins))
    assert list(reversed(margins)) == list(sorted(margins))

def mean_acc(result):
    tr_acc = np.mean([r['train_accuracy'] for r in result['splits']])
    te_acc = np.mean([r['test_accuracy'] for r in result['splits']])
    return tr_acc, te_acc

def test_boosting_for_smoke():
    n_examples = 1790
    X, y = digits_xy(n_examples)
    assert len(y) == n_examples

    n_rounds = 16
    n_features_per_round = 16

    def train_run(n_features, split_decisions, rseed=1):
        BI = BoostableDigits()
        config = dict(
            n_examples=n_examples,
            n_folds=5,
            feat_spec=dict(seed=rseed,
                n_features=n_features,
                scale=2.2),
            split_decisions=None, # gets clobbered by ctrl attachment
            save_svms=False       # good to test False sometimes
            )
        ctrl = Ctrl()
        ctrl.attachments['split_decisions'] = split_decisions
        result = BI.evaluate(config, ctrl)
        decisions = np.asarray(result['decisions'])
        print decisions.shape
        print 'mean abs decisions', abs(decisions).mean(),
        print 'mean margins', 1 - np.minimum(decisions * y, 1).mean(),
        tr_acc, te_acc = mean_acc(result)
        print 'train_accuracy', tr_acc,
        print 'test_accuracy', te_acc,
        print ''
        return decisions, tr_acc, te_acc

    print 'Training jointly'
    _, joint_tr_acc, joint_te_acc = train_run(
            n_rounds * n_features_per_round, None)

    print 'Training one round'
    _, one_tr_acc, one_te_acc = train_run(
            n_features_per_round, None)

    # -- train in rounds
    print 'Training in rounds'
    split_decisions = None
    for round_ii in range(n_rounds):
        foo = train_run(n_features_per_round, split_decisions, round_ii)
        split_decisions, tr_acc, te_acc = foo

    # assert that round-training and joint training are both way better than
    # training just one
    assert joint_tr_acc > 95
    assert joint_te_acc > 90
    assert one_tr_acc < 70
    assert one_te_acc < 70
    assert tr_acc > 90
    assert te_acc > 88


def test_random_search_boosting():
    n_examples = 1790
    X, y = digits_xy(n_examples)
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
                config = dict(
                    n_examples=n_examples,
                    n_folds=5,
                    feat_spec=dict(
                            seed=int(rstate.randint(2**31)),
                            n_features=int(rstate.randint(2, 32)),
                            scale=float(np.exp(rstate.randn())),
                        ),
                    split_decisions=None, # gets clobbered by ctrl attachment
                    save_svms=False       # good to test False sometimes
                    )
                ctrl = Ctrl()
                if selected:
                    split_decisions = np.asarray(selected[-1]['decisions'])
                else:
                    split_decisions = None
                ctrl.attachments['split_decisions'] = split_decisions

                result = BI.evaluate(config, ctrl)
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
                    continue
            else:
                selected.append(best)

            split_decisions = np.asarray(best['decisions'])
            print 'mean margins', 1 - np.minimum(split_decisions * y, 1).mean(),
            round_tr_acc, round_te_acc = mean_acc(best)
            print 'train_accuracy', round_tr_acc,
            print 'test_accuracy', round_te_acc,
            print ''
        return selected

    r1 = foo(1)
    r2 = foo(2)
    r5 = foo(5)

    assert r1[-1]['loss'] > r2[-1]['loss']
    assert r2[-1]['loss'] > r5[-1]['loss']


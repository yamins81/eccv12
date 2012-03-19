

class CVPRTopBandit(plugins.Bandit):
    param_gen = dict(
            slm=model_params.cvpr_top,
            comparison='mult',
            preproc={'global_normalize':1},
            )

data_root = '/home/bergstra/cvs/eccv12/eccv12/tests/data/'

def test_cvprtop_img_features():
    saved = np.load(os.path.join(data_root, 'fg11_features0-4.npy'))

    imgs = plugins.get_images('float32', None)  #--larray
    desc = copy.deepcopy(model_params.cvpr_top,
                memo={
                    id(model_params.null): None,
                    },)
    feat_fn = SLMFunction(desc, imgs.shape[1:])
    print saved[0].shape
    print feat_fn(imgs[0]).shape
    for ii, jj in (0, 0), (1, 1), (2, 3):
        assert np.allclose(saved[ii], feat_fn(imgs[jj]))

def test_cvprtop_features_all():
    filename='features_df448700aa91cef4c8bc666c75c393776a210177_0.dat'
    saved = np.memmap(os.path.join(data_root, filename),
            dtype='float32', mode='r',
            shape=(4992, 16, 16, 256))

    desc = copy.deepcopy(model_params.cvpr_top,
                memo={
                    id(model_params.null): None,
                    },)
    image_features = plugins.slm_memmap(
                desc=desc,
                X=plugins.get_images('float32', preproc=None),
                name='cvprtop_features_all_img_feat')
    vpairs_train = plugins.verification_pairs('DevTrain')
    vpairs_test = plugins.verification_pairs('DevTest')

    train_X, train_y = plugins.pairs_memmap(vpairs_train, image_features, 'mult', 'wtf_train')
    test_X, test_y =  plugins.pairs_memmap(vpairs_test , image_features, 'mult', 'wtf_test')

    # -- evaluate the whole set of pairs
    train_X = np.asarray(train_X)
    test_X = np.asarray(test_X)

    # -- check that there are 4992 valid entries in the image_features memmap
    #    and that our features match the saved ones

    print np.sum(image_features._valid)
    assert np.sum(image_features._valid) == 4992
    jj = 0
    for ii in range(4992):
        if image_features._valid[ii]:
            assert np.allclose(image_features._data[ii], saved[jj])
            jj += 1

    # -- check that our pair features match the saved ones
    saved_train_pairs_X = np.memmap(
            os.path.join(data_root,
                'train_pairs_df448700aa91cef4c8bc666c75c393776a210177.dat'),
            dtype='float32', mode='r',
            shape=(2200, 65536))
    saved_test_pairs_X = np.memmap(
            os.path.join(data_root,
                'test_pairs_df448700aa91cef4c8bc666c75c393776a210177.dat'),
            dtype='float32', mode='r',
            shape=(1000, 65536))

    assert np.allclose(train_X, saved_train_pairs_X)
    assert np.allclose(test_X, saved_test_pairs_X)

    train_d = np.zeros(len(train_y), dtype='float32')
    test_d  = np.zeros(len(test_y),  dtype='float32')

    train_Xyd_n, test_Xyd_n = plugins.normalize_Xcols(
        (train_X, train_y, train_d,),
        (test_X, test_y, test_d,))

    svm = plugins.train_svm(train_Xyd_n, l2_regularization=1e-3)

    new_d_train = plugins.svm_decisions(svm, train_Xyd_n)
    new_d_test = plugins.svm_decisions(svm, test_Xyd_n)

    result = plugins.result_binary_classifier_stats(
            train_Xyd_n,
            test_Xyd_n,
            new_d_train,
            new_d_test,
            result={})
    print 'Train_accuracy', result['train_accuracy']
    print 'Test accuracy', result['test_accuracy']
    print 'loss', result['loss'], np.sqrt(result['loss'] * (1 -
        result['loss']) / (len(test_y) - 1))

def test_cvprtop():
    bandit = CVPRTopBandit()
    config = bandit.template.sample(1)
    ctrl = hyperopt.Ctrl()
    result = bandit.evaluate(config, ctrl)

    print result['train_accuracy']
    print result['test_accuracy']
    print result['loss']
    # 100.0
    # 81.4
    # 0.186


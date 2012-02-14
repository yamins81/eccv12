from hyperopt.mongoexp import MongoJobs, MongoExperiment, as_mongo_str

DEFAULT_MONGO_WORKDIR = None
DEFAULT_MONGO_MONGO_OPTS = 'localhost/hyperopt'


class MongoMixin(object):
    def init_experiment(self, *args, **kwargs):
        workdir = getattr(self, 'workdir', DEFAULT_MONGO_WORKDIR)
        mongo_opts = getattr(self, 'mongo_opts', DEFAULT_MONGO_MONGO_OPTS)
        return init_mongo_exp(self.bandit_algo_class,
                              self.bandit_class,
                              bandit_argv=args,
                              bandit_kwargs=kwargs,
                              workdir=workdir,
                              mongo_opts=mongo_opts)


def init_mongo_exp(algo_name,
                   bandit_name,
                   bandit_argv=(),
                   bandit_kwargs=None,
                   algo_argv=(),
                   algo_kwargs=None,
                   workdir=None,
                   clear_existing=False,
                   force_lock=False,
                   mongo_opts='localhost/hyperopt'):

    ###XXX:  OK so this is ripped off from the code in hyperopt.mongoexp
    ###Perhaps it would be useful to modulized this functionality in a 
    ###new function in hyperopt.mongoexp, and then just call it here?

    if bandit_kwargs is None:
        bandit_kwargs = {}
    if algo_kwargs is None:
        algo_kwargs = {}
    if workdir is None:
        workdir = os.path.expanduser('~/.hyperopt.workdir')
    utils = hyperopt.utils
    bandit = utils.json_call(bandit_name, bandit_argv, bandit_kwargs)
    algo = utils.json_call(algo_name, (bandit,) + algo_argv, algo_kwargs)
    bandit_argfile_text = cPickle.dumps((bandit_argv, bandit_kwargs))
    algo_argfile_text = cPickle.dumps((algo_argv, algo_kwargs))    
    m = hashlib.md5()  ###XXX: why do we use md5 and not sha1? 
    m.update(bandit_argfile_text)
    m.update(algo_argfile_text)
    exp_key = '%s/%s[arghash:%s]' % (bandit_name, algo_name, m.hexdigest())
    del m
    worker_cmd = ('driver_attachment', exp_key)
    mj = MongoJobs.new_from_connection_str(as_mongo_str(mongo_opts) + '/jobs')
    experiment = MongoExperiment(bandit_algo=algo,
                                 mongo_handle=mj,
                                 workdir=workdir,
                                 exp_key=exp_key,
                                 cmd=worker_cmd)
    experiment.ddoc_get()  
    # XXX: this is bad, better to check what bandit_tuple is already there
    #      and assert that it matches if something is already there
    experiment.ddoc_attach_bandit_tuple(bandit_name,
                                        bandit_argv,
                                        bandit_kwargs)
    if clear_existing:
        print >> sys.stdout, "Are you sure you want to delete",
        print >> sys.stdout, ("all %i jobs with exp_key: '%s' ?"
                % (mj.jobs.find({'exp_key':exp_key}).count(),
                    str(exp_key)))
        print >> sys.stdout, '(y/n)'
        y, n = 'y', 'n'
        if input() != 'y':
            print >> sys.stdout, "aborting"
            del self
            return 1
        experiment.ddoc_lock(force=force_lock)
        experiment.clear_from_db()
        experiment.ddoc_get()
        experiment.ddoc_attach_bandit_tuple(bandit_name,
                                            bandit_argv,
                                            bandit_kwargs)
    return experiment


class ParallelMongoExperiment(ParallelExperiment, MongoMixin):
    pass


class BoostedMongoExperiment(BoostedExperiment, MongoMixin):
    def run_exp(self, exp, N):
        exp.run(N, block_until_done=True)

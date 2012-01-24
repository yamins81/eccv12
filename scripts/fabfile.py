import os

from fabric.api import cd
from fabric.api import env
from fabric.api import local
from fabric.api import run
from fabric.api import settings
from fabric.contrib.project import rsync_project

env.hosts = []
env.hosts.append('jbergstra@honeybadger.rowland.org:12122')


def push_skdata(dataset):
    """
    Copy a .skdata/`dataset` to the remote's .skdata/ folder
    """
    print ("Pushing %s dataset to %s" % (dataset, env.host_string,))
    run("mkdir -p ~/.skdata/%s" % dataset)
    rsync_project(remote_dir="~/.skdata",
            local_dir="~/.skdata/%s" % dataset,)


def rgit_branch(root):
    with cd(root):
        for line in run("git branch").split('\n'):
            if '*' in line:
                return line.split()[1]


def rgit_pull(name, gitsrc, add_pythonpath, branch=None):
    """
    Pull or clone the given branch of gitsrc into ~/cvs/`name`.
    """
    CVS = "~/cvs"  # don't expand locally
    TARGET = os.path.join(CVS, name)
    clone = True
    with settings(warn_only=True):
        if run("test -d %s" % TARGET).succeeded:
            clone = False
    if clone:
        run('mkdir -p %s' % CVS)
        with cd(CVS):
            run("git clone %(gitsrc)s %(name)s" % locals())
            if branch is not None and branch != rgit_branch(name):
                with cd(name):
                    run("git checkout -b %(branch)s origin/%(branch)s"
                            % locals())

        if add_pythonpath:
            PYTHONPATH = os.path.join(CVS, 'PYTHONPATH')
            run('mkdir -p %s' % PYTHONPATH)
            run('ln -s %s %s' % (
                os.path.join(TARGET, add_pythonpath),
                os.path.join(PYTHONPATH, add_pythonpath)))
    else:
        with cd(TARGET):
            run("git pull")



def rgit_pull_all():
    for r in repos:
        rgit_pull(**r)


#####################################################################
#####################################################################


def _git_repo(name, gitsrc, add_pythonpath=None, branch=None):
    if add_pythonpath is None:
        add_pythonpath=name
    _git_repo.repos.append(dict(name=name,
        gitsrc=gitsrc,
        add_pythonpath=add_pythonpath,
        branch=branch))
_git_repo.repos = []


_git_repo('Theano',
        'git://github.com/Theano/Theano.git',
        add_pythonpath='theano')


_git_repo('hyperopt',
        'git://github.com/jaberg/hyperopt.git',)


_git_repo('MonteTheano',
        'git://github.com/jaberg/MonteTheano.git',
        add_pythonpath='montetheano')


_git_repo('scikit-data',
        'git://github.com/jaberg/scikit-data.git',
        add_pythonpath='skdata')


_git_repo('eccv12',
        'git@github.com:nsf-ri-ubicv/boosted_hyperopt_eccv12.git',
        add_pythonpath='eccv12')


_git_repo('asgd',
        'git://github.com/jaberg/asgd.git')


_git_repo('genson',
        'git://github.com/yamins81/genson.git',
        branch='feature/hyperopt')


_git_repo('pythor3',
        'git@github.com:nsf-ri-ubicv/pythor3.git',
        branch='develop')


_git_repo('fbconv3-gcg',
        'git@github.com:jaberg/fbconv3-gcg.git',
        add_pythonpath="",
        branch='bandit')

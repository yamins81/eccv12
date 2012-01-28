import os

from fabric.api import cd
from fabric.api import env
from fabric.api import local
from fabric.api import run
from fabric.api import settings
from fabric.contrib.project import rsync_project

env.hosts = []
env.hosts.append('jbergstra@honeybadger.rowland.org:12122')

def baserun(cmd):
    return run(cmd)

def push_skdata(dataset):
    """
    Copy a .skdata/`dataset` to the remote's .skdata/ folder
    """
    print ("Pushing %s dataset to %s" % (dataset, env.host_string,))
    run("mkdir -p ~/.skdata/%s" % dataset)
    rsync_project(remote_dir="~/.skdata",
            local_dir="~/.skdata/%s" % dataset,)


def get_rgit_branch():
    for line in run("git branch").split('\n'):
        if '*' in line:
            return line.split()[1]


def rgit_pull(name=None):
    """
    Retrieve all of the development code used by the cvpr/eccv experiment.
    """
    if name is None:
        for r in _git_repo.repos:
            _rgit_pull(**r)
    else:
        for r in _git_repo.repos:
            if r['name'] == name:
                _rgit_pull(**r)


def install_dotfiles():
    with cd('cvs/dotfiles'):
        run('./install.sh')


def setup_venv():
    with settings(warn_only=True):
        if run('touch .virtualenv/base/bin/activate').failed:
            with settings(warn_only=False):
                run('mkdir .virtualenv')
                run('virtualenv .virtualenv/base')
                run('echo "source .virtualenv/base/bin/activate" >> .bashrc')


def setup_python_base():
    """
    Install non-standard python libraries into the virtual env that is set up
    by setup_venv()
    """
    simple_packages = ['pymongo',
            'bson',
            'lockfile',
            'pyparsing',
            'codepy',
            'PIL',
            ]

    for pkg in simple_packages:
        with settings(warn_only=True):
            if baserun('python -c "import %s"' % pkg).failed:
                with settings(warn_only=False):
                    baserun('pip install --user %s' % pkg)

    # install ordereddict for genson
    odict_test1 = 'python -c "import collections; collections.OrderedDict"'
    odict_test2 = 'python -c "import ordereddict"'
    with settings(warn_only=True):
        if baserun(odict_test1).failed and baserun(odict_test2).failed:
            with settings(warn_only=False):
                baserun('pip install --user -vUI ordereddict')


def setup_all():
    rgit_pull_all()
    setup_venv()
    setup_python_base()


def add_cvs_to_PYTHONPATH():
    paths = run(". .bashrc ; echo '$PYTHONPATH'").split(':')
    print paths
    for path in paths:
        if path.endswith('cvs/PYTHONPATH'):
            return
    run ("echo export 'PYTHONPATH=~/cvs/PYTHONPATH:$PYTHONPATH' >> .bashrc")


#####################################################################
#####################################################################

def _rgit_pull(name, gitsrc, add_pythonpath, branch=None):
    """
    Pull or clone the given branch of gitsrc into ~/cvs/`name`.
    """
    CVS = "~/cvs"  # don't expand locally
    clone = True
    join = os.path.join
    with settings(warn_only=True):
        if run("test -d %s" % join(CVS, name)).succeeded:
            clone = False
    if clone:
        run('mkdir -p %s' % CVS)
        with cd(CVS):
            run("git clone %(gitsrc)s %(name)s" % locals())
    else:
        with cd(join(CVS, name)):
            run("git pull")

    with cd(join(CVS, name)):
        if branch is not None and branch != get_rgit_branch():
            run("pwd")
            run("git checkout -b %(branch)s origin/%(branch)s" % locals())

    PYTHONPATH = join(CVS, 'PYTHONPATH')
    if add_pythonpath:
        run('mkdir -p %s' % PYTHONPATH)
        run('rm -f %s' % join(PYTHONPATH, add_pythonpath))
        run('ln -s %s %s' % (
            join(CVS, name, add_pythonpath),
            join(PYTHONPATH, add_pythonpath)))



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


_git_repo('thoreano',
        'git@github.com:jaberg/thoreano.git')


_git_repo('pythor3',
        'git@github.com:nsf-ri-ubicv/pythor3.git',
        branch='develop')


_git_repo('fbconv3-gcg',
        'git@github.com:jaberg/fbconv3-gcg.git',
        add_pythonpath="",
        branch='bandit')


_git_repo('dotfiles',
    'git@github.com:jaberg/dotfiles.git',
    add_pythonpath="")


_git_repo('hyperopt_cvpr2012', 'git@github.com:jaberg/hyperopt_cvpr2012.git',
        add_pythonpath='')

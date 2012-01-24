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
    return run('source .virtualenv/base/bin/activate; ' + cmd)

def push_skdata(dataset):
    """
    Copy a .skdata/`dataset` to the remote's .skdata/ folder
    """
    print ("Pushing %s dataset to %s" % (dataset, env.host_string,))
    run("mkdir -p ~/.skdata/%s" % dataset)
    rsync_project(remote_dir="~/.skdata",
            local_dir="~/.skdata/%s" % dataset,)


def get_rgit_branch(root):
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
            if branch is not None and branch != get_rgit_branch(name):
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
    """
    Retrieve all of the development code used by the cvpr/eccv experiment.
    """
    for r in _git_repo.repos:
        rgit_pull(**r)


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
    setup_venv()
    simple_packages = ['bson',
            'lockfile',
            'pyparsing',
            'codepy',
            'PIL',
            ]

    for pkg in simple_packages:
        with settings(warn_only=True):
            if baserun('python -c "import %s"' % pkg).failed:
                with settings(warn_only=False):
                    baserun('pip install %s' % pkg)

    # install ordereddict for genson
    odict_test1 = 'python -c "import collections; collections.OrderedDict"'
    odict_test2 = 'python -c "import ordereddict"'
    with settings(warn_only=True):
        if baserun(odict_test1).failed and baserun(odict_test2).failed:
            with settings(warn_only=False):
                baserun('pip install -vUI ordereddict')


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


_git_repo('dotfiles',
    'git@github.com:jaberg/dotfiles.git',
    add_pythonpath="")

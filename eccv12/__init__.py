import inspect
import os
import subprocess

# 
# -- This is some hacky shit to make the bandit behave nicely when qsub'd
#    on honeybadger.
#
def get_module_dir():
    return os.path.dirname(inspect.getfile(inspect.currentframe())) # script direct

# -- hacky stuff to configure theano prior to import when run via qsub
if 'THEANO_FLAGS' not in os.environ:
    require_gpu = False
    proc = subprocess.Popen(
        [os.path.join(get_module_dir(), 'hb_gpu_hack.sh')],
        stdout=subprocess.PIPE,
        #stderr=subprocess.PIPE,
        )
    my_gpu = proc.communicate()[0]
    if my_gpu:
        NEW_THEANO_FLAGS = ['device=gpu%i' % int(my_gpu)]
        require_gpu = True
    else:
        NEW_THEANO_FLAGS = []
    if os.path.exists('/scratch_local'):
        user = os.environ['USER']
        NEW_THEANO_FLAGS += ['base_compiledir=/scratch_local/' + user + '/eccv12.theano']
    os.environ['THEANO_FLAGS'] = ','.join(NEW_THEANO_FLAGS)

    print 'N.B. HACKING IN env["THEANO_FLAGS"] =', os.environ['THEANO_FLAGS']

    if require_gpu:
        # -- this is designed to crash on cluster machines, so that it does
        #    not waste everybody's time by running in slo-mo
        import theano
        import numpy as np
        testvar = theano.shared(np.ones(2, dtype='float32'))
        got_gpu = isinstance(testvar.type, theano.sandbox.cuda.CudaNdarrayType)
        assert got_gpu, 'failed to load CUDA'
        del testvar


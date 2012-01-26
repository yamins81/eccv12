import cPickle
import numpy as np

library = {}

def son(obj, *args, **kwargs):
    if isinstance(obj, (int, str, list, tuple, float)):
        return obj
    return obj.son(*args, **kwargs)


class NodeFactory(object):
    def __init__(self, f, name, call_w_scope):
        self.f = f
        self.call_w_scope = call_w_scope
        if name is None:
            self.name = f.__name__
        else:
            self.name = name
        library[self.name] = self

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def son(self, *args, **kwargs):
        return dict(
            _fn_=self.name,
            call_w_scope=self.call_w_scope,
            args=args,
            kwargs=kwargs)


def register(name=None, call_w_scope=False):
    def deco(f):
        return NodeFactory(f, name, call_w_scope=call_w_scope)
    return deco


def fson_print(node, prefix=0):
    if isinstance(node, dict) and '_fn_' in node:
        args = node.get('args', [])
        kwargs = node.get('kwargs', {})
        if args or kwargs:
            print ' ' * prefix + node['_fn_'] + '('
            for a in node['args']:
                fson_print(a, prefix + 4)
            for k, v in node['kwargs'].items():
                print ' ' * prefix + '  ' + k + ' ='
                fson_print(v, prefix + 4)
            print ' ' * prefix + '  )'
        else:
            print ' ' * prefix + node['_fn_'] + '()'
    else:
        print ' ' * prefix + str(node)


def fson_eval(node, memo={}, scope={}):
    if isinstance(node, dict) and '_fn_' in node:
        args = [fson_eval(a, memo, scope) for a in node.get('args', [])]
        kwargs = dict([(k, fson_eval(v, memo, scope))
                        for (k, v) in node.get('kwargs', {}).items()])
        if node['call_w_scope']:
            kwargs.update(scope)
        rval = library[node['_fn_']](*args, **kwargs)
        memo[id(node)] = rval
        return rval
    else:
        memo[id(node)] = node
        return node


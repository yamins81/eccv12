if 0:
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
        if type(name) == type(register):
            raise TypeError('register must be *called* to be a decorator')
        def deco(f):
            return NodeFactory(f, name, call_w_scope=call_w_scope)
        return deco


    def fson_count_nodes(node, counts):
        counts.setdefault(id(node), 0)
        counts[id(node)] += 1
        if isinstance(node, dict) and '_fn_' in node:
            for a in node.get('args', []):
                fson_count_nodes(a, counts)
            for v in node.get('kwargs', {}).values():
                fson_count_nodes(v, counts)
        return counts


    def fson_print_helper(node, prefix, memo, counts):
        if id(node) in memo:
            print ' ' * prefix + memo[id(node)]
            return

        if isinstance(node, dict) and '_fn_' in node:
            args = node.get('args', [])
            kwargs = node.get('kwargs', {})
            if args or kwargs:
                print ' ' * prefix + node['_fn_'] + '('
                for a in node['args']:
                    fson_print(a, prefix + 4, memo)
                for k, v in node['kwargs'].items():
                    print ' ' * prefix + '  ' + k + ' ='
                    fson_print(v, prefix + 4, memo)
                print ' ' * prefix + '  )'
            else:
                print ' ' * prefix + node['_fn_'] + '()'
        else:
            print ' ' * prefix + str(node)


    def fson_print(node, prefix=0, memo=None):
        if memo is None:
            memo = {}
        counts = fson_count_nodes(node, {})
        return fson_print_helper(node, prefix, memo, counts)


    def fson_eval(node, memo=None, scope=None):
        if memo is None:
            memo = {}
        if scope is None:
            scope = {}
        try:
            return memo[id(node)]
        except KeyError:
            pass

        if isinstance(node, dict) and '_fn_' in node:
            args = [fson_eval(a, memo, scope) for a in node.get('args', [])]
            kwargs = dict([(k, fson_eval(v, memo, scope))
                            for (k, v) in node.get('kwargs', {}).items()])
            if node['call_w_scope']:
                kwargs['scope'] = scope
            rval = library[node['_fn_']](*args, **kwargs)
            memo[id(node)] = rval
            return rval
        else:
            memo[id(node)] = node
            return node

    class fson_function(object):
        def __init__(self, node):
            self.node = node

        def __call__(self, **kwargs):
            return fson_eval(self.node, memo={}, scope=kwargs)

    @register()
    def run_all(*args, **kwargs):
        """
        Putting this at the top-level of an fson program will force the evaluation
        of all arguments, but return None.
        """
        return None

from paddle.fluid.wrapped_decorator import signature_safe_contextmanager, wrap_decorator
from paddle.fluid.framework import in_dygraph_mode

@signature_safe_contextmanager
def name_scope(prefix=None):

    # TODO(panyx0718): Only [0-9a-z].
    # in dygraph we don't need namescope since it will cause mem leak
    if in_dygraph_mode():
        yield
    else:
        assert prefix, "namescope prefix can not be empty."
        global _name_scope
        _name_scope = _name_scope.child(prefix)
        try:
            yield
        finally:
            _name_scope = _name_scope.parent()

def control_dependencies(control_inputs):
    print("control_dependencies not implemented.")
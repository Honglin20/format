import contextvars
from typing import List

import torch.nn as nn

_module_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "quant_module_stack", default=[]
)


def get_layer_name() -> str:
    stack = _module_stack.get()
    return ".".join(stack) if stack else ""


def _make_pre_hook(name: str):
    def _pre(module, inp):
        stack = list(_module_stack.get())
        stack.append(name)
        _module_stack.set(stack)
    return _pre


def _make_post_hook(name: str):
    def _post(module, inp, out):
        stack = list(_module_stack.get())
        if stack and stack[-1] == name:
            stack.pop()
            _module_stack.set(stack)
    return _post


def install_stack_hooks(model: nn.Module) -> List:
    """Register forward pre/post hooks on all named sub-modules. Returns handles."""
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        handles.append(module.register_forward_pre_hook(_make_pre_hook(name)))
        handles.append(module.register_forward_hook(_make_post_hook(name)))
    return handles


def remove_stack_hooks(handles: List) -> None:
    for h in handles:
        h.remove()

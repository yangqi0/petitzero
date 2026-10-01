"""The M6B validated F-M execution profile. Author: Yang Qi."""
from contextlib import contextmanager
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

def metadata():
    return dict(attention='sdpa',flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled(),cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),math_reduced_precision=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),autocast=torch.is_autocast_enabled('cuda'))

@contextmanager
def fm_profile():
    prior=metadata()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    try:
        with sdpa_kernel(SDPBackend.MATH),torch.autocast('cuda',enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(prior['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32=prior['matmul_tf32']
        torch.backends.cudnn.allow_tf32=prior['cudnn_tf32']
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(prior['math_reduced_precision'])

def assert_fm():
    m=metadata()
    assert m['math'] and not any(m[k] for k in ['flash','efficient','cudnn_sdp','matmul_tf32','cudnn_tf32','math_reduced_precision','autocast'])
    assert m['float32_matmul_precision']=='highest'

import pytest
import imaging.deconv_sharpen as mod

CASES = [
    ("calibrated_psf", (1,)),
    ("mirror_pad", (1,)),
    ("overlap_window", (1,)),
    ("global_normalize", (1,)),
    ("soft_clip", (1,)),
    ("determinism", (1,)),
    ("deconvolve", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("fft_block", (1,)),
    ("metrics", (1,)),
    ("ringing_check", (1,)),
    ("noise_ratio", (1,)),
    ("output", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None

"""Keep controller fixtures independent of the optional upstream vocabularies."""
import os

import pytest

from physedit import compact_prompt


@pytest.fixture(autouse=True)
def synthetic_controller_token_counts(request, monkeypatch):
    if request.module.__name__ not in ('test_generation_pipeline', 'test_inversion_inputs'):
        return
    if os.environ.get('PHYSEDIT_TOKENIZERS'):
        return

    def count(text):
        size = len(text.split()) + 1
        return dict(ltx=size, cogvideox=size, wan=size)

    monkeypatch.setattr(compact_prompt, 'counts', count)

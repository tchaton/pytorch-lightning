# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
import os

import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from pytorch_lightning.callbacks import EarlyStopping
from tests.base import EvalModelTemplate
from pytorch_lightning.core import memory
from pytorch_lightning.trainer import Trainer
from tests.backends import ddp_model
from tests.utilities.dist import call_training_script


@pytest.mark.parametrize('cli_args', [
    pytest.param('--max_epochs 1 --num_nodes 2 --gpus 1 --distributed_backend ddp'),
])
def test_multi_gpu_model_ddp_fit_only(tmpdir, cli_args):
    # call the script
    std, err = call_training_script(ddp_model, cli_args, 'fit', tmpdir, timeout=120)

    # load the results of the script
    result_path = os.path.join(tmpdir, 'ddp.result')
    result = torch.load(result_path)

    # verify the file wrote the expected outputs
    assert result['status'] == 'complete'
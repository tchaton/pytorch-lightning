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
import sys
import os
ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

import pytest
import torch
from unittest import mock
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from tests.base.boring_model import *


@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1', reason="test should be run outside of pytest")
def test_logging_sync_dist_true_cpu(tmpdir):
    """
    Tests to ensure that the sync_dist flag works with CPU (should just return the original value)
    """
    fake_result = 1

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log('foo', torch.tensor(fake_result), on_step=False, on_epoch=True, sync_dist=True)
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log('bar', torch.tensor(fake_result), on_step=False, on_epoch=True, sync_dist=True)
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        weights_summary=None,
        accelerator="ddp",
        gpus=1,
        num_nodes=2,
    )
    trainer.fit(model)

    assert trainer.logged_metrics['foo'] == fake_result
    assert trainer.logged_metrics['bar'] == fake_result

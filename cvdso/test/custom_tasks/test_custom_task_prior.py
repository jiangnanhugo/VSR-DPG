import pytest
from cvdso.test.generate_test_data import CONFIG_TRAINING_OVERRIDE
from cvdso.test.test_core import model


def test_custom_task(model):
    model.config_task = {
        "task_type": "cvdso.test.custom_tasks.custom_task_prior:CustomTask",
        "param" : "test"
    }
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()


def test_custom_prior(model):
    model.config_prior["cvdso.test.custom_tasks.custom_task_prior:CustomPrior"] = {
        "param" : "test",
         "on" : True
    }
    model.config_training.update(CONFIG_TRAINING_OVERRIDE)
    model.train()

from types import SimpleNamespace

from pipeline_error_training import resources


def test_monitor_child_environment_is_stable_during_process_updates(monkeypatch):
    key = 'PIPELINE_ERROR_RESOURCE_TEST'
    monkeypatch.setenv(key, 'before-launch')

    def launch(command, **kwargs):
        child_environment = kwargs['env']
        monkeypatch.setenv(key, 'updated-during-launch')
        assert child_environment[key]=='before-launch'
        assert command[0]=='nvidia-smi' and kwargs['check']
        return SimpleNamespace(stdout='5120, 75\n')

    monkeypatch.setattr(resources.subprocess, 'run', launch)
    assert resources.gpu_snapshot()==dict(total_gib=5., utilization_percent=75.)

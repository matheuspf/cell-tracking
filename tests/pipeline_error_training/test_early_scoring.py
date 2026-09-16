from pipeline_error_training.early_scoring import inventory_ready


def test_early_scoring_requires_every_complete_clip_and_serialized_artifact(tmp_path):
    rows = [{'dataset':'first'},{'dataset':'second'}]
    arm = 'A10'
    assert not inventory_ready(arm,[],tmp_path)
    for row in rows:
        name = row['dataset']
        marker = tmp_path/'final_point_inference'/arm/name/'complete.json'
        marker.parent.mkdir(parents=True)
        marker.write_text('{}')
        prediction = tmp_path/'predictions'/arm/name
        prediction.parent.mkdir(parents=True,exist_ok=True)
        for suffix in ['.npz','.json','.csv','.guard.json']:
            prediction.with_suffix(suffix).write_text('fixture')
    assert inventory_ready(arm,rows,tmp_path)
    (tmp_path/'predictions'/arm/'second.csv').unlink()
    assert not inventory_ready(arm,rows,tmp_path)
    (tmp_path/'predictions'/arm/'second.csv').write_text('fixture')
    (tmp_path/'final_point_inference'/arm/'second/complete.json').unlink()
    assert not inventory_ready(arm,rows,tmp_path)


def test_division_completion_uses_the_whole_clip_matrix_marker(tmp_path):
    rows = [{'dataset':'clip'}]
    arm = 'D20_temporal_replacement'
    prediction = tmp_path/'predictions'/arm/'clip'
    prediction.parent.mkdir(parents=True)
    for suffix in ['.npz','.json','.csv','.guard.json']:
        prediction.with_suffix(suffix).write_text('fixture')
    assert not inventory_ready(arm,rows,tmp_path)
    marker = tmp_path/'final_inference/clip/complete.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    assert inventory_ready(arm,rows,tmp_path)

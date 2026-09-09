import pandas as pd
import pytest

from strong_tracker_v3.report import choose


def points(variants):
    rows=[]
    for name,(pooled,a,b) in variants.items():
        for embryo,delta in [('pooled',pooled),('44b6',a),('6bba',b)]:
            rows.append(dict(variant=name,embryo=embryo,score=.9342063149703403+delta,
                             delta_v2=delta))
    return pd.DataFrame(rows)


def test_pooled_gain_cannot_override_embryo_regret_or_invalid_control():
    data=points({'incumbent':(0,0,0),'A_good':(.001,.0001,.0011),
                 'D_bad':(.005,-.0001,.006),'R_image_local':(.01,.01,.01),
                 'C_oracle':(.02,.02,.02),'historical_v1':(.03,.03,.03)})
    assert choose(data)==('A_good',['A_good'])


def test_retain_v2_without_a_qualified_gain_and_tolerate_only_scoring_noise():
    data=points({'incumbent':(0,0,0),'A_negative':(-.001,-.001,-.001)})
    assert choose(data)==('incumbent',[])
    data=points({'incumbent':(0,0,0),'A_noise':(.0001,-.5e-10,.0002)})
    assert choose(data)==('A_noise',['A_noise'])


def test_cli_budget_includes_historical_control(tmp_path):
    from strong_tracker_v3.__main__ import enforce_variant_budget
    from strong_tracker_v3.common import write_json
    from strong_tracker_v3.context import RunContext
    ctx=RunContext.default(out=tmp_path)
    existing=['historical_v1',*[f'v{i}' for i in range(30)]]
    write_json(tmp_path/'rounds/first.json',{'variants':dict.fromkeys(existing,{})})
    enforce_variant_budget(ctx,['v31'])
    with pytest.raises(ValueError,match='historical'):
        enforce_variant_budget(ctx,['v31','v32'])

import runpy


def test_overview_review_selection_is_deterministic_and_diverse():
    module=runpy.run_path('reports/maritime-expansion-20261007/earthview-adapter/prepare_review.py')
    rows=[dict(sample_id=f'source-{i:03}',port=dict(id=str(i%20),country=str(i%5),size='L'),
        acquisition_identity=[f'2022-10-{i%28+1:02}T00:00:00Z'],
        seam_diagnostics=dict(boundaries=[dict(ratio=3 if i<12 else 1)])) for i in range(120)]
    first=module['select'](rows,80);second=module['select'](list(reversed(rows)),80)
    assert [row['sample_id'] for row in first]==[row['sample_id'] for row in second]
    assert len(first)==80 and len({row['sample_id'] for row in first})==80
    assert {row['port']['country'] for row in first}=={str(i) for i in range(5)}
    assert {row['port']['id'] for row in first}=={str(i) for i in range(20)}
    assert set(f'source-{i:03}' for i in range(12))<={row['sample_id'] for row in first}

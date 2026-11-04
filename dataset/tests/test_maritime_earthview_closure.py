import io,json,tracemalloc
import pytest
from scripts.maritime_earthview_closure import manifest_items

def test_manifest_stream_exact_rows_and_scalars_with_small_chunks():
 document=dict(candidate_count=2,candidates=[dict(sample_id='one',nested=dict(x=['a',2])),dict(sample_id='two')],rejections=[dict(sample_id='three')],config=dict(a=True))
 entries=list(manifest_items(io.StringIO(json.dumps(document,sort_keys=True)),chunk_size=7))
 assert [v for k,v in entries if k=='candidates']==document['candidates']
 assert [v for k,v in entries if k=='rejections']==document['rejections']
 assert dict((k,v) for k,v in entries if k not in ('candidates','rejections'))==dict(candidate_count=2,config=dict(a=True))

def test_manifest_stream_memory_does_not_retain1000_large_rows():
 payload=json.dumps(dict(candidates=[dict(sample_id=str(i),payload='x'*100000) for i in range(1000)],rejections=[]))
 source=io.StringIO(payload);tracemalloc.start();count=0
 for key,row in manifest_items(source):count+=key=='candidates'
 _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 assert count==1000 and peak<2000000

def test_manifest_stream_fails_closed_truncation_and_oversized_item():
 with pytest.raises(ValueError):list(manifest_items(io.StringIO('{"candidates":[{"x":1}')))
 with pytest.raises(ValueError):list(manifest_items(io.StringIO('{"candidates":[{"x":"'+('a'*200)+'"}]}'),chunk_size=20,max_item_chars=100))


def test_manifest_stream_scalar_exponent_across_chunk_and_empty_trailing_fail():
 assert dict(manifest_items(io.StringIO('{"number":1e-20}'),chunk_size=2))==dict(number=1e-20)
 with pytest.raises(ValueError):list(manifest_items(io.StringIO('{}trailing')))

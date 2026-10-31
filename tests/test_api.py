"""Offline API tests: mocks are execution fixtures, never empirical evidence."""
import json
from unittest.mock import Mock
import pytest
import requests

from energy_pilot.api import Client,PRICES


def response(payload,code=200,headers=None):
    r=Mock(status_code=code,headers=headers or {},url='https://example.invalid/fixture')
    r.json.return_value=payload
    if code>=400: r.raise_for_status.side_effect=requests.HTTPError(str(code))
    return r


def test_cache_and_offline(tmp_path,monkeypatch):
    monkeypatch.setattr('energy_pilot.api.time.sleep',lambda _:None)
    c=Client(tmp_path)
    c.session.get=Mock(return_value=response({'records':[{'x':1}]}))
    assert c.get('fixture',{'a':1})=={'records':[{'x':1}]}
    assert c.session.get.call_count==1
    cached=Client(tmp_path,offline=True)
    assert cached.get('fixture',{'a':1})=={'records':[{'x':1}]}
    envelope=json.loads(next(tmp_path.glob('*.json')).read_text())
    assert envelope['parameters']=={'a':1} and 'downloaded_utc' in envelope
    with pytest.raises(FileNotFoundError,match='Offline cache miss'): cached.get('other')


def test_retry_after_respected_in_chunks(tmp_path,monkeypatch):
    waits=[]
    monkeypatch.setattr('energy_pilot.api.time.sleep',waits.append)
    c=Client(tmp_path)
    c.session.get=Mock(side_effect=[response({},429,{'Retry-After':'78'}),response({'ok':1})])
    assert c.get('fixture')=={'ok':1}
    assert waits[:3]==[30,30,18] and c.session.get.call_count==2


def test_retry_exhaustion_and_no_cached_failure(tmp_path,monkeypatch):
    monkeypatch.setattr('energy_pilot.api.time.sleep',lambda _:None)
    c=Client(tmp_path)
    c.session.get=Mock(return_value=response({},503))
    with pytest.raises(requests.HTTPError): c.get('fixture')
    assert c.session.get.call_count==4 and not list(tmp_path.glob('*.json'))


def test_unexpected_schema_and_truncation_fail(tmp_path):
    c=Client(tmp_path)
    c.get=Mock(return_value={'columns':[{'dbColumn':'HourUTC'}]})
    with pytest.raises(ValueError,match='schema'): c.metadata(PRICES)
    c.get=Mock(return_value={'records':[{'HourUTC':'2024-01-01','PriceArea':'DK1','SpotPriceEUR':1}], 'total':2})
    with pytest.raises(ValueError,match='truncated'): c.window(PRICES,'2024-01-01','2024-01-02')

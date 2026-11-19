import json
from pathlib import Path


def html(payload):
    # Embedding is safe even for a fictional/malformed horse name containing HTML.
    data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    return Path(__file__).with_name('dashboard.html').read_text(encoding='utf-8').replace('__DUEL_DATA__',data)

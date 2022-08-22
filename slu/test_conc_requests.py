import time
import json
import random
import asyncio
from pprint import pprint

import aiohttp
from aiohttp.client_exceptions import ServerDisconnectedError
import pandas as pd


async def conc_slu_requests(url, data, l=30, force_close=True):
    c = aiohttp.TCPConnector(limit=l, force_close=force_close)
    async with aiohttp.ClientSession(url, connector=c) as session:
        intents = await get_intents(session, data)
    return intents


async def get_intents(session, data):
    return await asyncio.gather(*[get_intent(session, payload) for payload in data])


async def get_intent(session, payload):
    try:
        start = time.perf_counter()
        async with session.post("/predict/hi/name/", json=payload) as res:
            if res.status == 200:
                res = await res.json()
                end = time.perf_counter()
                if "response" not in res:
                    print(res)
                    return "exception"
                else:
                    pprint(res["response"]["intents"])
                    return (res["response"]["intents"][0]["name"], end - start)
            else:
                err = await res.text()
                end = time.perf_counter()
                print(err)
                return ("error", end - start)
    except (ServerDisconnectedError, asyncio.TimeoutError) as e:
        end = time.perf_counter()
        print(e)
        return ("error", end - start)


def test(fpath, n=-1, n_conn=30, force_close_conns=True):
    df = pd.read_csv(fpath)
    df.input = df.input.apply(json.loads)
    data = df.to_dict(orient="records")
    data = [d["input"] for d in data]
    random.shuffle(data)
    data = data[:n] if n > -1 else data
    loop = asyncio.get_event_loop()
    print(f"request for {len(data)}")
    start = time.perf_counter()
    results = loop.run_until_complete(
                conc_slu_requests("http://localhost:8090",
                data,
                l=n_conn,
                force_close=force_close_conns))
    end = time.perf_counter()
    print(f"Processed {len(results)} in {end - start}s")
    return results

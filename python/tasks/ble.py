import bleak
import pprint

async def f1():
    devices = await bleak.BleakScanner.discover()
    return devices

async def f2(device, timeout=None):
    if timeout is None:
        timeout = 1.0

    assert isinstance(timeout, float) and timeout >= 1e-8

    p = await bleak.BleakClient(
        device,
        timeout=timeout,
    ).__aenter__()
    return p

async def f3(client):
    t1 = [
        dict(
            service=o.__dict__,
            characteristics=[
                o2.__dict__
                for o2 in o.characteristics
            ]
        )
        for o in client.services
    ]
    return t1

async def f5(name=None):
    t2 = []

    attempt = 0
    while True:
        t1 = await f1()
        pprint.pprint([o.__dict__ for o in t1])

        if not name is None:
            assert isinstance(name, str)
            t5 = {
                i : o.details[0].name()
                for i, o in enumerate(t1)
            }

            t2.extend(
                [
                    t1[k]
                    for k, v in t5.items()
                    if isinstance(v, str) and v.lower() == name
                ]
            )

        if len(t2) > 0:
            break

        attempt += 1
        print('\rattempt #%d' % attempt, end='')

    return t2

async def f4(timeout=None):
    t2 = await f5(name='watch fit')

    if len(t2) == 0:
        print('not found')
        return

    t3 = None
    try:
        t3 = await f2(t2[0], timeout=timeout)
        t4 = await f3(t3)
        pprint.pprint(t4)
    finally:
        if not t3 is None:
            await t3.disconnect()

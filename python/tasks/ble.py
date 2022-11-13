import bleak
import inspect
import traceback
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

async def f5(
    name_check=None,
):
    t2 = []

    attempt = 0

    while True:
        t1 = await f1()
        pprint.pprint([o.__dict__ for o in t1])

        if not name_check is None:
            assert inspect.isfunction(name_check)

            t5 = {
                i : o.details[0].name()
                for i, o in enumerate(t1)
            }

            t2.extend(
                [
                    t1[k]
                    for k, v in t5.items()
                    if isinstance(v, str) and name_check(v)
                ]
            )
        else:
            t2.extend(t1)

        if len(t2) > 0:
            break

        attempt += 1
        print('\rattempt #%d' % attempt, end='')

    return t2

async def f4(
    timeout=None,
    characteristics=None,
    operations=None,
    name_check=None,
):
    if isinstance(name_check, str):
        assert name_check in [
            'watch fit',
        ]
        name_check2 = lambda current_name: name_check.lower() in current_name.lower()
    else:
        name_check2 = name_check

    assert not name_check2 is None

    if characteristics is None:
        characteristics = [
            '0000ffd1-0000-1000-8000-00805f9b34fb',
        ]

    t2 = await f5(
        name_check=name_check2,
    )

    if len(t2) == 0:
        print('not found')
        return

    t3 = None
    try:
        t3 = await f2(t2[0], timeout=timeout)
        t4 = await f3(t3)
        pprint.pprint(t4)

        if not operations is None and inspect.isfunction(operations):
            await operations(
                client=t3,
                t4=t4,
            )
        else:
            t6 = {}
            for o in characteristics:
                try:
                    t7 = await t3.read_gatt_char(o)
                except Exception as exception:
                    print(traceback.format_exc())
                    t7 = None
                t6[o] = t7
            pprint.pprint(t6)
    finally:
        if not t3 is None:
            await t3.disconnect()

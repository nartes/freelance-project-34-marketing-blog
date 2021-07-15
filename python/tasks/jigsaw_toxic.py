import pprint
import requests
import pyquery


def kernel_1_sample_scrap():
    t8 = []
    for t7 in [
        'https://dev.to/kunaal438/media-query-everything-you-need-for-responsive-design-b8g',
    ]:
        with requests.get(
            t7,
        ) as p:
            t1 = p.content.decode('utf-8')
        t2 = pyquery.PyQuery(t1)
        t3 = t2('.comment__content')
        t6 = []
        for o in t3:
            t4 = pyquery.PyQuery(o)
            t5 = t4('.comment__header > a').attr['href']
            t6.append(
                dict(
                    author=t5,
                )
            )

        #pprint.pprint(t3)
        pprint.pprint(t6)
        t8.append(
            dict(
                article=t7,
                comments=t6,
            )
        )

    pprint.pprint(t8)

    return dict(
        t1=t1,
        t2=t2,
        t3=t3,
        t6=t6,
        t8=t8,
    )

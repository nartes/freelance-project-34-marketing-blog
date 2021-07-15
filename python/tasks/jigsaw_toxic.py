import pprint
import requests
import pyquery


def kernel_1_sample_scrap():
    with requests.get(
        'https://dev.to/kunaal438/media-query-everything-you-need-for-responsive-design-b8g',
    ) as p:
        t1 = p.content.decode('utf-8')
    t2 = pyquery.PyQuery(t1)
    t3 = t2('.comment_content')
    pprint.pprint(t3)

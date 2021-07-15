def kernel_1_sample_scrap():
    with requests.get(
        'https://dev.to/kunaal438/media-query-everything-you-need-for-responsive-design-b8g',
    ) as p:
        t1 = p.content
    pprint.pprint(t1)

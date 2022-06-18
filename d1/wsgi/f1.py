with io.open('~/p1/p1/1.json', 'r') as f:
    t1 = json.load(f)

t2 = sys.argv[1:]
pprint.pprint(t1)
pprint.pprint(t2)
pprint.pprint([(a, b) for a, b in zip(sorted(t2), sorted(t1['t17']))])

#t3 = t2[:-2] + ['-d', '@-']
t3 = t1['t17'] + ['-o', '1.dat']
#t3 = t4['t17']
print(t3)

p = None
try:
    with subprocess.Popen(t3, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as p:
        p.stdin.write(t1['t3'].encode('utf-8'))
        p.stdin.flush()
        p.stdin.close()
        p.wait(20)
except:
    p.terminate()


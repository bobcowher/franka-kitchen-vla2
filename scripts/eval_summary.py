"""Success rate per checkpoint and task from results_*.jsonl, with Wilson 95% intervals."""
import collections, glob, json, math, sys
d = sys.argv[1]
c = collections.defaultdict(lambda: [0, 0])
for path in glob.glob(f"{d}/*.jsonl"):
    for line in open(path):
        r = json.loads(line)
        c[(r["label"], r["task"])][0] += r["success"]
        c[(r["label"], r["task"])][1] += 1
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z*z/n; mid = (p + z*z/(2*n)) / den
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return mid - h, mid + h
key = lambda l: (int(l.split("@")[0]), l.split("@")[1])
tasks = ["microwave", "hinge cabinet", "slide cabinet", "top burner"]
print("ckpt".ljust(12) + "".join(t.ljust(20) for t in tasks))
for label in sorted({k[0] for k in c}, key=key):
    row = label.ljust(12)
    for t in tasks:
        k, n = c.get((label, t), [0, 0])
        row += (f"{k}/{n} [{wilson(k,n)[0]:.2f}-{wilson(k,n)[1]:.2f}]" if n else "-").ljust(20)
    print(row)

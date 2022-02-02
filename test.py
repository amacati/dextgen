import random
import time
from tqdm import tqdm


neps = 100
p = tqdm(total=neps, desc="whatever", position=0, leave=False)
d = tqdm(total=0, position=1, bar_format="{desc}", leave=False)

for _ in range(neps):
    p.update()
    d.set_description(f"Loss: {random.random():.2f}")
    time.sleep(0.1)
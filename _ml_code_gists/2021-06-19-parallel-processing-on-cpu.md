---
title: "Parallel Processing on CPU"
last_modified_at: 2021-06-17T21:30:02-05:00
excerpt: How can we parallize out function on CPU?
categories:
  - Blogs
collections:
  - ml_code_gists
---

```python
from concurrent.futures import ProcessPoolExecutor
```

```python
def parallel(func, arr, max_workers=4):
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results
```
# Dataloader caching

This project aims to cache dataset or part of it in memory to speedup training.

If your PyTorch training project is suffering from low disk reading, it tries to cache dataset or part of it in memory.

Possible use cases:
- Iterate all  dataset   ```__getitem__()``` with limit size. Set num_worker=0. Minimal and easy. See [example/mnist.py](example/mnist.py)
- Cache image files (jpeg compressed) and store it PyTorch distributed cache.
- Cache dataset between machines (*TODO*)



# Installation 
`pip install --upgrade git+https://github.com/chroneus/DataloaderCache.git`


# Minimal example benchmark

MNIST 4V100 total training before:  **2 min 17 sec**

after replacing dataloader(dataset) with CachedDataset(dataset,preload=True)  **42 sec**



# Dataloader caching

This project aims to cache dataset or part of it in memory to speedup training.

If your PyTorch training project is suffering from low disk reading, it tries to cache dataset or part of it in memory.

Possible use cases:
- Iterate all  dataset   ```__getitem__()``` with limit size . Minimal and easy. See [example/mnist.py](example/mnist.py)
- Cache image files (jpeg compressed) and store it PyTorch distributed cache.
- Cache dataset between machines (TODO)



## installation 
`pip install --upgrade git+https://github.com/chroneus/DataloaderCache.git`

## usage

# minimal example :
MNIST 4V100 total training before:  2min17sec
after replacing dataloader(dataset) with CachedDataset(dataset,preload=True)  42sec



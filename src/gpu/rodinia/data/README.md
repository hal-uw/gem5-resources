# Rodinia input data

The benchmarks read their inputs from this directory (`../../data/<benchmark>/...`
relative to `hip/<benchmark>/`). The data is not in the repository; download the
Rodinia 3.1 data set and unpack it here.

## Download

- Rodinia 3.1 download page: http://lava.cs.virginia.edu/Rodinia/download_links.htm
  (the data comes with the full `rodinia_3.1` package)
- Data-only mirror (from [yuhc/gpu-rodinia](https://github.com/yuhc/gpu-rodinia)):
  https://www.dropbox.com/s/cc6cozpboht3mtu/rodinia-3.1-data.tar.gz
  (396 MB compressed, 1.1 GB unpacked;
  SHA-256 `b90994d5208ec5a0a133dfb9ab7928a1e8a16741503a91d212884b9e4fce8cd8`)

```sh
cd data
curl -L -o rodinia-3.1-data.tar.gz 'https://www.dropbox.com/s/cc6cozpboht3mtu/rodinia-3.1-data.tar.gz?dl=1'
echo 'b90994d5208ec5a0a133dfb9ab7928a1e8a16741503a91d212884b9e4fce8cd8  rodinia-3.1-data.tar.gz' | sha256sum -c
tar xzf rodinia-3.1-data.tar.gz --strip-components=1   # the archive's top directory is rodinia-data/
rm rodinia-3.1-data.tar.gz
```

The archive has a directory per benchmark (`bfs/`, `b+tree/`, `cfd/`, `dwt2d/`,
`gaussian/`, `heartwall/`, `hotspot/`, `hotspot3D/`, `huffman/`, `lud/`, `myocyte/`,
`nn/`, `srad/`, ...), including the `hotspot3D` and `huffman` inputs and the full
50 MB `heartwall/test.avi`. Truncated copies of `test.avi` hold only 7 readable
frames, so heartwall fails with the default 20 frames.

## Files the default runs use

About 100 MB of the set:

| Benchmark | Input |
|---|---|
| bfs | `bfs/graph1MW_6.txt` |
| b+tree | `b+tree/mil.txt`, `b+tree/command.txt` |
| cfd (euler3d, pre_euler3d and their _double versions) | `cfd/fvcorr.domn.097K` |
| dwt2d | `dwt2d/192.bmp` |
| gaussian | `gaussian/matrix4.txt` |
| heartwall | `heartwall/test.avi` |
| hotspot | `hotspot/temp_512`, `hotspot/power_512` |
| hotspot3D | `hotspot3D/power_512x8`, `hotspot3D/temp_512x8` |
| huffman | `huffman/test1024_H2.206587175259.in` |
| myocyte | `myocyte/` |
| nn | `nn/cane4_*.db` (listed in `hip/nn/filelist_4`) |
| srad_v1 | `srad/image.pgm` |

backprop, lavaMD, lud (`-s`), nw, particlefilter, pathfinder, srad_v2 and
streamcluster generate their inputs and need no files.

# 128k feasibility na M5 / 32 GB (SuperGemma MLX)

## Stručný závěr

**Skutečný single-window kontext 128k tokenů není na tomto stroji realistický** (32 GB unified memory) pro tento model.  
Model to architektonicky podporuje, ale paměťově to nevychází.

## Empirické měření (`max-kv-size` probe)

Artefakty:
- `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/artifacts/benchmarks/20260414T140214Z`
- `/Users/satan/side/experiments/supergemma-dflash-ddtree-mlx/artifacts/benchmarks/20260414T140321Z`

Výsledek:
- ověřeno bez pádu až do `max-kv-size=32768`
- první fail v tomto rozsahu **nenastal**

| kv size | status | prompt tok/s | generation tok/s | peak memory (GB) |
|---:|---|---:|---:|---:|
| 512 | ok | 19.207 | 46.825 | 14.279 |
| 1024 | ok | 20.907 | 48.328 | 14.279 |
| 2048 | ok | 28.774 | 48.725 | 14.279 |
| 4096 | ok | 25.963 | 44.414 | 14.279 |
| 8192 | ok | 25.041 | 46.959 | 14.279 |
| 16384 | ok | 26.562 | 47.621 | 14.279 |
| 32768 | ok | 24.923 | 48.137 | 14.279 |

### Důležitá interpretace

Tyto probe běhy mají krátký prompt (`~27` tokenů), takže **netlačí KV cache do skutečné velikosti**.  
`max-kv-size` je strop, ne okamžitě alokovaná plná cache.

## Paměťový model KV cache

Konfigurační hodnoty modelu (`config.json` → `text_config`):
- `num_hidden_layers = 30`
- `num_key_value_heads = 8`
- `head_dim = 256`
- `dtype = bfloat16` (2 bajty)

KV bajty na token:

`30 * 8 * 256 * 2 (K+V) * 2 bytes = 245760 bytes ≈ 0.234 MiB/token`

KV cache odhad:

| Kontext (tokeny) | KV cache (GiB) |
|---:|---:|
| 4,096 | 0.938 |
| 8,192 | 1.875 |
| 16,384 | 3.750 |
| 32,768 | 7.500 |
| 65,536 | 15.000 |
| 128,000 | 29.297 |

Váhy modelu (`model.safetensors.index.json`):  
`total_size = 14,200,055,868 bytes ≈ 13.225 GiB`

Celkový hrubý odhad (váhy + KV, bez dalších overheadů):
- 65,536 tokenů: `13.225 + 15.000 ≈ 28.225 GiB` (už velmi těsné)
- 128,000 tokenů: `13.225 + 29.297 ≈ 42.522 GiB` (**nad 32 GB**)

## Závěr pro 128k

1. **Single-window 128k na tomto stroji ne** (paměťově mimo limit).
2. Praktická cesta je **long-context pipeline (chunking + map-reduce)** místo jedné 128k inference.

# binet · 引文网络双向爬取工具

`binet` (BiblioNet-Bidirectional Crawler) 从一批种子 DOI 出发，**双向**扩张引文网络，
构建有向引文图 `i→j`（表示 *i 引用 j*），作为后续文献耦合 / 共被引 / PageRank 等
特征工程的图基础。

实现完全对齐《双向引文网络需求文档 v1.0》(FR-1~6, AC-1~6)。

---

## 1. 安装依赖

仅依赖 `requests` 与 `tqdm`（项目 `requirements.txt` 已包含）：

```bash
pip install requests tqdm
```

---

## 2. 快速开始

```bash
# 默认行为：max_depth=1，双向，max_papers=2000（AC-2）
python -m binet.main --seeds 10.18653/v1/2025.acl-long.907

# 仅向后（references），两层，限 100 节点
python -m binet.main --seeds 10.1103/PhysRevLett.127.136101 \
    --direction backward --max-depth 2 --max-papers 100

# 从文件读取种子（每行一个 DOI，或一个 JSON 数组）
python -m binet.main --seed-file seeds.txt

# 中断后断点续传（AC-4）
python -m binet.main --resume --output-dir binet/output

# 额外导出下游图格式
python -m binet.main --seeds 10.xxxx/yyyy --export-edgelist --export-graphml
```

---

## 3. 数据源策略（方向非对称）

| 方向 | 主源 | Fallback |
|---|---|---|
| 向后 references | OpenAlex (`referenced_works`) | CrossRef (`reference[].DOI`) |
| 向前 citations (cited-by) | OpenAlex (`filter=cites:`) | Semantic Scholar (`/citations`) |

- 所有源实现统一契约 `CitationSource`（`binet/src/base.py`，§5.2）。
- 不支持的方向（如 CrossRef 的 cited-by）会抛 `NotSupportedError`，**不会静默返回错方向数据**。
- 数据源级联：主源失败/无数据时按链自动回退；命中次数记入报告。
- 请求头携带 polite-pool 邮箱（默认 `dedsecjohn@163.com`），可用 `--email` 覆盖。

---

## 4. 主要参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--seeds` | — | 种子 DOI（可多个） |
| `--seed-file` | — | 种子文件（每行一个 DOI 或 JSON 数组） |
| `--max-depth` | `1` | 从种子向外的最大跳数 |
| `--backward-depth` / `--forward-depth` | =max-depth | 分方向深度（FR-2.3） |
| `--direction` | `both` | `backward` / `forward` / `both` |
| `--max-papers` | `2000` | 节点总数硬熔断（FR-2.4） |
| `--reference-sources` | `openalex crossref` | references 回退链 |
| `--citation-sources` | `openalex semantic_scholar` | citations 回退链 |
| `--delay-min` / `--delay-max` | `0.5` / `1.0` | 随机请求间隔（FR-5.3） |
| `--max-retries` | `3` | 指数退避重试次数（FR-5.1） |
| `--checkpoint-every` | `50` | 每处理 N 个节点存档（FR-5.4） |
| `--output-dir` | `binet/output` | 输出目录 |
| `--resume` | off | 从 checkpoint 续传 |
| `--export-edgelist` / `--export-graphml` | off | 导出下游图格式 |
| `--verbose` | off | 详细日志 |

---

## 5. 输出产物（`output/`）

| 文件 | 说明 |
|---|---|
| `citation_network.json` | 主产物：`metadata` + `nodes` + `edges`（§4.1） |
| `failed_dois.txt` | 失败 DOI 及原因 |
| `checkpoint.json` | 断点状态（成功完成后自动清理） |
| `crawl.log` | 运行日志 |
| `crawl_report.json` | 覆盖度审计：种子平均语料内入度等（AC-6） |
| `edgelist.csv` / `*.graphml` | 可选下游图格式（§4.3，为 networkx/PageRank 铺路） |

`citation_network.json` 结构示例：

```json
{
  "metadata": {
    "seed_dois": ["..."], "max_depth": 1, "backward_depth": 1, "forward_depth": 1,
    "max_papers": 2000, "total_nodes": 0, "total_edges": 0, "failed_count": 0,
    "dropped_edges_no_doi": 0,
    "data_sources": {"references": ["openalex","crossref"],
                     "citations": ["openalex","semantic_scholar"]},
    "created_at": "ISO-8601",
    "status": "completed | max_papers_reached | interrupted"
  },
  "nodes": [{"doi": "10.x/y", "title": "...", "depth": 0, "is_seed": true}],
  "edges": [{"source_doi": "10.a/b", "target_doi": "10.c/d"}]
}
```

---

## 6. 模块结构

```
binet/
  config.py        # 配置与默认值 (FR-6)
  doi_utils.py     # DOI 归一化 (FR-3.2)
  errors.py        # NotSupportedError / DeterministicFailure
  models.py        # PaperMeta / QueueItem / NodeRecord / Edge (§5.3)
  http_client.py   # 重试退避 + 速率控制 + polite-pool (FR-5)
  src/
    base.py        # CitationSource 契约 (§5.2)
    openalex.py    # 主源：references + cited-by
    crossref.py    # references
    semantic.py    # citations fallback
  crawler.py       # 双向 BFS 主引擎 (FR-1~4)
  checkpoint.py    # 断点续传 (FR-5.4)
  serialize.py     # 输出 JSON / edgelist / GraphML (§4.1/4.3)
  report.py        # 覆盖度审计 (AC-6)
  main.py          # 命令行入口 (FR-6)
  output/          # 输出目录
```


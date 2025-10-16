# arXiv Daily Summaries（cs.SD / eess.AS）

本项目通过 GitHub Actions 每日抓取 arXiv 指定分类的论文，
并将英文摘要翻译为中文，自动生成站点文章（Jekyll / GitHub Pages）。

## 目录结构

```
.
├─ .github/workflows/
│  └─ arxiv-daily.yml
├─ site/
│  ├─ _config.yml
│  ├─ _posts/
│  └─ index.md
├─ scripts/
│  └─ fetch_and_summarize.py
├─ data/
│  └─ seen_ids.json
├─ requirements.txt
└─ README.md
```

## 运行前准备

- 需要一个可用的 OpenAI API Key：`OPENAI_API_KEY`
- Python 3.10+（建议 3.11）

## 本地运行

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=YOUR_KEY
# 可选覆写（默认 cs.SD,eess.AS / 20 篇 / 26 小时窗口）
export OPENAI_MODEL=gpt-4o-mini
export ARXIV_CATEGORIES="cs.SD,eess.AS"
export MAX_PAPERS="10"
export TIME_WINDOW_HOURS="26"

python scripts/fetch_and_summarize.py
```

生成的 Markdown 会写入 `site/_posts/YYYY-MM-DD-arxiv.md`。

## 部署到 GitHub Pages


```
git add -A
git commit -m "update"
git push --force-with-lease origin main
```


1. 在仓库 Settings → Secrets and variables → Actions 新增：
   - `OPENAI_API_KEY`
2. 在 Settings → Pages：Build and deployment → Source 选择 **GitHub Actions**。
3. 之后工作流会每日运行并部署静态站点。

## 注意

- 只翻译 arXiv ATOM feed 的摘要文本，不读取 PDF。
- 为控制成本，可调整 `MAX_PAPERS`、模型与提示词长度。

## 过滤设置（可选）

- `STRICT_PRIMARY_ONLY`（默认 1）：只保留主分类为 `cs.SD` 或 `eess.AS` 的论文。
- `KEYWORDS_INCLUDE`：逗号分隔关键词，命中任意一个（标题/摘要）才保留，例如：
  ```bash
  export STRICT_PRIMARY_ONLY="1"
  export KEYWORDS_INCLUDE="speech,asr,tts,voice,audio,spoken"
  ```
- 调试：
  - `DEBUG_LIST_CATEGORIES=1` 仅打印 `aid/primary/cats/title` 与过滤结果，不翻译；
  - `DRY_RUN=1` 正常生成但不调用 OpenAI（写入占位文本）；
  - `RESET_SEEN=1` 忽略已见 ID 去重，便于复跑。
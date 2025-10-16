import os, json, time, re
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
import feedparser
from urllib.parse import urlencode, quote_plus

# OpenAI SDK (>=1.0)
from openai import OpenAI
# 延迟初始化，避免调试模式需要 OPENAI_API_KEY
client = None

# 环境变量参数（可在 GitHub Actions 中覆写）
CATEGORIES = os.environ.get("ARXIV_CATEGORIES", "cs.SD,eess.AS")
MAX_PAPERS = int(os.environ.get("MAX_PAPERS", "20"))     # 每天最多翻译多少篇，防止token开销过大
TIME_WINDOW_HOURS = int(os.environ.get("TIME_WINDOW_HOURS", "24"))  # 最近多少小时内更新的论文
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")   # 可换成你账户可用、性价比高的模型
OUT_DIR = "site/_posts"
DB_PATH = "data/seen_ids.json"
RESET_SEEN = os.environ.get("RESET_SEEN", "0").lower() in ("1", "true", "yes")
DEBUG_LIST_CATEGORIES = os.environ.get("DEBUG_LIST_CATEGORIES", "0").lower() in ("1", "true", "yes")
DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes")
# 仅主分类过滤（默认开启，确保主分类为 cs.SD 或 eess.AS）
STRICT_PRIMARY_ONLY = 0
# 关键词包含（可选）：逗号分隔；命中任意一个（标题/摘要）才保留
KEYWORDS_INCLUDE = [
    kw.strip().lower() for kw in os.environ.get("KEYWORDS_INCLUDE", "").split(",") if kw.strip()
]

# arXiv API：按提交时间倒序
def build_arxiv_query(categories: str) -> str:
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    query = " OR ".join([f"cat:{c}" for c in cats])
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": 100,
    }
    return "http://export.arxiv.org/api/query?" + urlencode(params, quote_via=quote_plus)

def load_seen_ids() -> set:
    if not os.path.exists(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, "w") as f:
            json.dump([], f)
    with open(DB_PATH, "r") as f:
        return set(json.load(f))

def save_seen_ids(ids_set: set) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(sorted(list(ids_set)), f, ensure_ascii=False, indent=2)

def arxiv_id_from_link(link: str) -> str:
    # e.g., "http://arxiv.org/abs/2501.01234v1" -> "2501.01234v1"
    m = re.search(r'arxiv\.org/abs/([^/]+)$', link)
    return m.group(1) if m else link

def fetch_recent_entries():
    url = build_arxiv_query(CATEGORIES)
    feed = feedparser.parse(url)
    if feed.bozo:
        raise RuntimeError(f"Feed parse error: {feed.bozo_exception}")
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(hours=TIME_WINDOW_HOURS)
    entries = []
    for e in feed.entries:
        updated_raw = getattr(e, "updated", getattr(e, "published", ""))
        if not updated_raw:
            continue
        updated = dtparser.parse(updated_raw)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        if updated >= window_start:
            entries.append(e)
    return entries

def extract_categories(e) -> list:
    cats = []
    for t in getattr(e, "tags", []):
        term = getattr(t, "term", None)
        if term is None and isinstance(t, dict):
            term = t.get("term")
        if term:
            cats.append(term)
    # 兼容主分类字段
    apc = getattr(e, "arxiv_primary_category", None)
    if apc:
        primary_term = apc.get("term") if isinstance(apc, dict) else getattr(apc, "term", None)
        if primary_term and primary_term not in cats:
            cats.append(primary_term)
    return cats

def get_primary_category(e) -> str:
    apc = getattr(e, "arxiv_primary_category", None)
    if apc:
        return apc.get("term") if isinstance(apc, dict) else getattr(apc, "term", None)
    return None

def translate_abstract_to_zh(title, authors, summary, categories, link, pdf_link) -> str:
    # 仅进行中文翻译与简要结构化整理，不需要读取 PDF
    global client
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        client = OpenAI(api_key=api_key)
    sys = "你是资深学术翻译助理。请把英文摘要翻译成高质量中文."
    user = f"""
请将以下 arXiv 论文的英文摘要翻译成高质量中文：
1) 不复述题目；2) 语言准确精炼；3) 面向语音/音频/AI 研究者；

【基本信息】
- 标题：{title}
- 作者：{", ".join(authors)}
- 类别：{", ".join(categories)}
- 链接：{link}
- PDF：{pdf_link}

【英文摘要】
{summary}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    return resp.choices[0].message.content.strip()

def ensure_posts_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def write_daily_post(date_str: str, items: list, total_entries: int) -> str:
    """
    生成 Jekyll 博文：
    文件名形如：YYYY-MM-DD-arxiv.md
    """
    ensure_posts_dir()
    out_path = os.path.join(OUT_DIR, f"{date_str}-arxiv.md")
    lines = []
    lines.append("---")
    lines.append(f'layout: post')
    lines.append(f'title: "arXiv Daily – {date_str}"')
    lines.append("tags: [arxiv, cs.SD, eess.AS]")
    lines.append("---\n")
    lines.append(f"以下为 {date_str}（UTC±）窗口内更新的论文中文译文：")
    lines.append(f"- 时间窗口：最近 {TIME_WINDOW_HOURS} 小时")
    lines.append(f"- 抓取总数：{total_entries} 篇 | 本页显示：{len(items)} 篇（去重/过滤后）\n")

    for it in items:
        lines.append(f"## {it['title']}")
        lines.append(f"- **Authors**: {', '.join(it['authors'])}")
        lines.append(f"- **Categories**: {', '.join(it['categories'])}")
        lines.append(f"- **arXiv**: [{it['link']}]({it['link']})  |  **PDF**: [{it['pdf']}]({it['pdf']})")
        lines.append("")
        lines.append(it["summary_md"])  # 中文译文/要点
        lines.append("\n---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    return out_path

def main() -> None:
    seen = set() if RESET_SEEN else load_seen_ids()
    entries = fetch_recent_entries()
    total_entries_count = len(entries)
    # 允许的分类集合（与查询参数一致）
    allowed = {c.strip() for c in CATEGORIES.split(",") if c.strip()}
    if not entries:
        print("No recent entries.")
        return

    # 仅打印分类调试信息后退出
    if DEBUG_LIST_CATEGORIES:
        print(f"Fetched entries: {len(entries)} | Allowed: {sorted(list(allowed))} | STRICT_PRIMARY_ONLY={STRICT_PRIMARY_ONLY} | KEYWORDS_INCLUDE={KEYWORDS_INCLUDE}")
        kept = 0
        for e in entries:
            link = getattr(e, "link", "")
            aid = arxiv_id_from_link(link)
            title = getattr(e, "title", "").strip()
            cats = extract_categories(e)
            primary = get_primary_category(e)
            by_tags = any(cat in allowed for cat in cats)
            by_primary = (primary in allowed) if STRICT_PRIMARY_ONLY else by_tags
            if KEYWORDS_INCLUDE:
                text = f"{title}\n" + getattr(e, "summary", "")
                text_l = text.lower()
                by_keywords = any(kw in text_l for kw in KEYWORDS_INCLUDE)
            else:
                by_keywords = True
            by_keywords = True
            is_allowed = by_primary and by_keywords
            print(f"[allowed={is_allowed}] {aid} | primary={primary} | cats={cats} | by_tags={by_tags} by_primary={by_primary} by_keywords={by_keywords} | title={title}")
            if is_allowed:
                kept += 1
        print(f"Kept after filter: {kept}")
        return

    items = []
    count = 0
    for e in entries:
        link = getattr(e, "link", "")
        aid = arxiv_id_from_link(link)
        if aid in seen:
            continue
        title = e.title.strip()
        # authors
        authors = [a.name for a in getattr(e, "authors", [])] or []
        # categories（更稳健提取）
        cats = extract_categories(e)
        # 仅保留 cs.SD / eess.AS（主分类可选严格）
        primary = get_primary_category(e)
        allowed_by_tags = any(cat in allowed for cat in cats)
        allowed_by_primary = (primary in allowed) if STRICT_PRIMARY_ONLY else allowed_by_tags
        if not allowed_by_primary:
            print(f"Not allowed by primary: {aid} | primary={primary} | cats={cats} | by_tags={allowed_by_tags} by_primary={allowed_by_primary}")
            continue


        # abstract
        abstract = getattr(e, "summary", "").strip()
        # pdf link
        pdf = ""
        for l in getattr(e, "links", []):
            if l.get("type") == "application/pdf":
                pdf = l.get("href")
                break
        pdf = pdf or (link.replace("/abs/", "/pdf/") + ".pdf")

        if DRY_RUN:
            # 调试模式下不调用 OpenAI，直接记录占位文本
            md = "（调试）仅打印分类，不进行翻译。"
        else:
            # 调用 OpenAI 翻译（不读取 PDF）
            try:
                md = translate_abstract_to_zh(title, authors, abstract, cats, link, pdf)
            except Exception as ex:
                print(f"OpenAI translate failed for {aid}: {ex}")
                continue

        items.append({
            "id": aid,
            "title": title,
            "authors": authors,
            "categories": cats,
            "link": link,
            "pdf": pdf,
            "summary_md": md
        })
        seen.add(aid)
        count += 1
        if count >= MAX_PAPERS:
            break
        time.sleep(0.8)  # 轻微节流

    save_seen_ids(seen)

    if items:
        today = datetime.now(timezone.utc).date().isoformat()
        write_daily_post(today, items, total_entries_count)
    else:
        print("No new items to write.")

if __name__ == "__main__":
    main()



import os, json, time, re
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
import feedparser
from urllib.parse import urlencode, quote_plus
from zoneinfo import ZoneInfo
import requests
import shutil

import fitz  # pip install pymupdf

# OpenAI SDK (>=1.0)
from openai import OpenAI
# 延迟初始化，避免调试模式需要 API Key
client = None
client_provider = None

# 环境变量参数（可在 GitHub Actions 中覆写）
CATEGORIES = os.environ.get("ARXIV_CATEGORIES", "cs.SD,eess.AS")
MAX_PAPERS = int(20)     # 每天最多翻译多少篇，防止token开销过大
TIME_WINDOW_HOURS = int(os.environ.get("TIME_WINDOW_HOURS", "24"))  # 最近多少小时内更新的论文
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")   # 可换成你账户可用、性价比高的模型
OUT_DIR = "site/_posts"
DB_PATH = "data/seen_ids.json"
RESET_SEEN = os.environ.get("RESET_SEEN", "1").lower() in ("1", "true", "yes")
DEBUG_LIST_CATEGORIES = os.environ.get("DEBUG_LIST_CATEGORIES", "0").lower() in ("1", "true", "yes")
DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes")
# 仅主分类过滤（默认开启，确保主分类为 cs.SD 或 eess.AS）
STRICT_PRIMARY_ONLY = 0
# 关键词包含（可选）：逗号分隔；命中任意一个（标题/摘要）才保留
KEYWORDS_INCLUDE = 0

# PDF 解析与全文解读设置
PDF_DIR = os.environ.get("PDF_DIR", "data/pdfs")
ANALYSIS_MAX_CHARS = int(os.environ.get("ANALYSIS_MAX_CHARS", "100000"))
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "")  # 为空则回落到 MODEL 或 deepseek 默认

# LLM 提供方（openai / deepseek），deepseek 使用 OpenAI 兼容 API
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek").lower()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/beta")


def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        # "text"最干净；"blocks"保留块结构；"raw"更接近原始顺序；"xhtml"含标记
        texts.append(page.get_text("text"))
    full_text = "\n".join(texts)
    # 查找最后一次出现的 "references"（不区分大小写），并丢弃其后的内容（包含匹配本身）
    last_index = -1
    for match in re.finditer(r'(?i)\breferences\b', full_text):
        last_index = match.start()
    if last_index != -1:
        return full_text[:last_index].rstrip()
    return full_text


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

def compute_anchored_window():
    """返回以北京时间 08:50 为锚点的 24 小时窗口（昨日 08:50 → 今日 08:50）。
    end 锚点取“最近一次不超过当前时间的 08:50”。"""
    tz_cst = ZoneInfo("Asia/Shanghai")
    now_cst = datetime.now(tz_cst)
    anchor_today = now_cst.replace(hour=8, minute=50, second=0, microsecond=0)
    if now_cst >= anchor_today:
        end_cst = anchor_today
    else:
        end_cst = anchor_today - timedelta(days=1)
    start_cst = end_cst - timedelta(days=1)
    start_utc = start_cst.astimezone(timezone.utc)
    end_utc = end_cst.astimezone(timezone.utc)
    return start_utc, end_utc, start_cst, end_cst

def fetch_recent_entries():
    url = build_arxiv_query(CATEGORIES)
    feed = feedparser.parse(url)
    if feed.bozo:
        raise RuntimeError(f"Feed parse error: {feed.bozo_exception}")
    window_start_utc, window_end_utc, window_start_cst, window_end_cst = compute_anchored_window()
    entries = []
    for e in feed.entries:
        # 优先基于 published 过滤；缺失时退回 updated
        ts_raw = getattr(e, "published", getattr(e, "updated", ""))
        if not ts_raw:
            continue
        ts = dtparser.parse(ts_raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # 使用 [start, end) 半开区间，避免与隔天窗口重叠
        if window_start_utc <= ts < window_end_utc:
            entries.append(e)
    return entries, window_start_cst, window_end_cst

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

def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    # 把所有空白（含换行、制表）折叠为单个空格
    return re.sub(r"\s+", " ", str(text)).strip()

def ensure_client():
    global client, client_provider
    if client is not None:
        return
    if LLM_PROVIDER == "deepseek":
        # OpenAI 兼容客户端（base_url+api_key）
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        client_provider = "deepseek"
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        client = OpenAI(api_key=api_key)
        client_provider = "openai"

def translate_abstract_to_zh(title, authors, summary, categories, link, pdf_link) -> str:
    # 仅进行中文翻译与简要结构化整理，不需要读取 PDF
    ensure_client()
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
    model_name = MODEL if LLM_PROVIDER != "deepseek" else os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    return resp.choices[0].message.content.strip()

def analyze_fulltext(title: str, text: str) -> str:
    ensure_client()
    if not text:
        return "（无可用全文）"
    trimmed = text[:ANALYSIS_MAX_CHARS]
    sys = "你是资深学术助理，请基于提供的论文全文文本，用中文简洁、有条理地作答。"
    user = f"""
请基于下方全文内容，分别用条理清晰的短段落（限制字数）回答三个问题：
1) 研究背景与既有方法的问题（≤200字）。
2) 论文核心方法如何解决上述问题（≤600字）。
3) 在哪些任务上取得了怎样的效果（≤200字）。
注意：
- 用1/2/3分条输出, 在每个问题中可以使用 - 等列表符号梳理说明；
- 不要复述题目；
- 使用markdown格式输出。

【标题】{title}
【正文】
{trimmed}
"""
    model_name = (ANALYSIS_MODEL or MODEL)
    if LLM_PROVIDER == "deepseek":
        model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ]
    )
    return resp.choices[0].message.content.strip()

def ensure_posts_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def write_daily_post(date_str: str, items: list, total_entries: int, window_start_cst: datetime, window_end_cst: datetime) -> str:
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
    lines.append(f"以下为 {date_str}（CST，北京时间）窗口内更新的论文中文译文：")
    lines.append(f"- 时间窗口（锚点 08:50 CST）：{window_start_cst.strftime('%Y-%m-%d %H:%M')} — {window_end_cst.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 抓取总数：{total_entries} 篇 | 本页显示：{len(items)} 篇（去重/过滤后）\n")

    for it in items:
        lines.append(f"## {it['title']}")
        lines.append(f"- **Authors**: {', '.join(it['authors'])}")
        lines.append(f"- **Categories**: {', '.join(it['categories'])}")
        lines.append(f"- **arXiv**: [{it['link']}]({it['link']})")
        lines.append(f"- **PDF**: [{it['pdf']}]({it['pdf']})")
        lines.append("")
        lines.append(it["summary_md"])  # 中文译文/要点
        # 折叠的“详细解读”
        if it.get("analysis_md"):
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>详细解读</summary>")
            lines.append("")
            lines.append("<div markdown=\"1\">")
            lines.append("")
            lines.append(it["analysis_md"])  # 模型输出的三问三答（渲染为 Markdown）
            lines.append("</div>")
            lines.append("")
            lines.append("</details>")
        lines.append("\n---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")
    return out_path



def cleanup_pdfs_folder():
    try:
        if os.path.isdir(PDF_DIR):
            shutil.rmtree(PDF_DIR)
    except Exception as ex:
        print(f"Cleanup PDFs failed: {ex}")

def main() -> None:
    seen = set() if RESET_SEEN else load_seen_ids()
    entries, window_start_cst, window_end_cst = fetch_recent_entries()
    total_entries_count = len(entries)
    # 允许的分类集合（与查询参数一致）
    allowed = {c.strip() for c in CATEGORIES.split(",") if c.strip()}
    if not entries:
        print("No recent entries.")
        return

    # 仅打印分类调试信息后退出
    if DEBUG_LIST_CATEGORIES:
        print(f"Window CST: {window_start_cst.strftime('%Y-%m-%d %H:%M')} — {window_end_cst.strftime('%Y-%m-%d %H:%M')}")
        print(f"Fetched entries: {len(entries)} | Allowed: {sorted(list(allowed))} | STRICT_PRIMARY_ONLY={STRICT_PRIMARY_ONLY} | KEYWORDS_INCLUDE={KEYWORDS_INCLUDE}")
        kept = 0
        for e in entries:
            link = getattr(e, "link", "")
            aid = arxiv_id_from_link(link)
            title = normalize_whitespace(getattr(e, "title", ""))
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
    try:
        for e in entries:
            link = getattr(e, "link", "")
            aid = arxiv_id_from_link(link)
            if aid in seen:
                continue
            title = normalize_whitespace(e.title)
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
                # 调试模式下不调用 LLM，直接记录占位文本
                md = "（调试）仅打印分类，不进行翻译。"
                analysis_md = "（调试）未进行全文解读。"
            else:
                # 调用 LLM 翻译（不读取 PDF）
                try:
                    md = translate_abstract_to_zh(title, authors, abstract, cats, link, pdf)
                except Exception as ex:
                    print(f"OpenAI translate failed for {aid}: {ex}")
                    continue
                # 下载 PDF 并做全文解读（独立对话）
                analysis_md = ""
                try:
                    os.makedirs(PDF_DIR, exist_ok=True)
                    pdf_local_path = os.path.join(PDF_DIR, f"{aid}.pdf")
                    if not os.path.exists(pdf_local_path):
                        r = requests.get(pdf, timeout=60)
                        r.raise_for_status()
                        with open(pdf_local_path, "wb") as pf:
                            pf.write(r.content)
                    full_text = extract_text_pymupdf(pdf_local_path)
                    analysis_md = analyze_fulltext(title, full_text)
                except Exception as ex:
                    analysis_md = f"（全文解读失败：{ex}）"

            items.append({
                "id": aid,
                "title": title,
                "authors": authors,
                "categories": cats,
                "link": link,
                "pdf": pdf,
                "summary_md": md,
                "analysis_md": analysis_md
            })
            seen.add(aid)
            count += 1
            if count >= MAX_PAPERS:
                break
            time.sleep(0.8)  # 轻微节流
    finally:
        # 无论成功失败，都清理 PDF 目录
        cleanup_pdfs_folder()

    save_seen_ids(seen)

    if items:
        # 以窗口结束日（CST）命名当日文件
        date_str = window_end_cst.date().isoformat()
        write_daily_post(date_str, items, total_entries_count, window_start_cst, window_end_cst)
    else:
        print("No new items to write.")

if __name__ == "__main__":
    main()



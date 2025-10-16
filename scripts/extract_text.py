import fitz  # pip install pymupdf
import re

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

if __name__ == "__main__":
    text = extract_text_pymupdf("Moshi.pdf")
    # print(text)
    print(len(text))
    with open("Moshi.txt", "w") as f:
        f.write(text)

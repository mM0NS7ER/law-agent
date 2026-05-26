import re
import json
from docx import Document


ARTICLE_HEADING_RE = re.compile(r"(?m)^\s*第[一二三四五六七八九十百千万零〇0-9]+条(?!的)")
SUB_ARTICLE_RE = re.compile(r"^\s*第[一二三四五六七八九十百千万零〇0-9]+条之[一二三四五六七八九十百千万零〇0-9]+")


def convert_penal_code_docx_to_jsonl(docx_path, output_path):
    """
    将刑法.docx按条拆分为JSONL格式
    :param docx_path: 输入文件路径，如 '刑法.docx'
    :param output_path: 输出文件路径，如 '刑法.jsonl'
    """
    doc = Document(docx_path)
    full_text = []
    
    # 1. 读取所有段落文本
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            full_text.append(text)
    
    # 2. 合并全文并初步清理
    full_content = '\n'.join(full_text)
    full_content = re.sub(r'\n+', '\n', full_content)  # 合并多余换行
    
    # 3. 核心：按“第x条”的起始位置切分，避免 re.split 吞掉正文
    matches = list(ARTICLE_HEADING_RE.finditer(full_content))
    print(f"检测到疑似条标题数量: {len(matches)}")
    
    # 4. 清洗并构建JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        article_id = 0
        main_article_count = 0
        sub_article_count = 0
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_content)
            block = full_content[start:end].strip()
            if not block:
                continue

            # 以首次换行拆分 title/content；若无换行则尽量在条标题后按空格拆分
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            title = lines[0] if lines else match.group(0)
            content = "\n".join(lines[1:]).strip()
            if not content:
                content = block[len(title):].strip()

            # 有些数据源把整条挤在一行，至少保留该条原文，不再因空内容被丢弃
            if not content:
                content = block

            if SUB_ARTICLE_RE.match(title):
                sub_article_count += 1
            else:
                main_article_count += 1

            article_data = {
                "id": article_id,
                "title": title,
                "content": content,
                "metadata": {
                    "law": "中华人民共和国刑法",
                    "type": "条文"
                }
            }
            f.write(json.dumps(article_data, ensure_ascii=False) + '\n')
            article_id += 1
    
    print(
        f"转换完成！主条文 {main_article_count} 条，增补条(条之一/之二...) {sub_article_count} 条，"
        f"共 {article_id} 条，输出至 {output_path}"
    )

# 使用示例
if __name__ == "__main__":
    convert_penal_code_docx_to_jsonl('C:\\Users\\Lawrence\\Desktop\\AI_STUDY\\interview\\law-agent\\ttt.docx', 'C:\\Users\\Lawrence\\Desktop\\AI_STUDY\\interview\\law-agent\\data\\crimina_law_china.jsonl')
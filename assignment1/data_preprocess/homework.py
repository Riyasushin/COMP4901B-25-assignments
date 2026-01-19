import argparse
import re
import requests
import json
from utils import  read_warc_file, read_wet_file
from datasets import load_dataset
from typing import Set, Dict
import string

def retrieve_bad_words() -> set[str]:
    """Helper function - that reads a list of bad words from a file and returns them as a set.
    Returns:
        Set[str]: A set containing lowercase bad words.
    """
    with open('./bad_word_list.txt', 'r') as file:
        records = file.read().strip().split('\n')
        bad_words = [record.lower() for record in records]
        return set(bad_words)

# 移除 script/style 标签及内容
SCRIPT_STYLE_PAT = re.compile(r'<(script|style)[^>]*?>.*?</\1>', re.IGNORECASE | re.DOTALL)
# 移除所有 HTML 标签
TAG_PAT = re.compile(r'<[^>]+>')
# 匹配 img 标签的 alt 属性
IMG_ALT_PAT = re.compile(r'<img[^>]+alt=[\'"]([^\'"]+)[\'"][^>]*>', re.IGNORECASE)
# 匹配 a 标签的文本和 href
A_TAG_PAT = re.compile(r'<a[^>]+href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
# 合并连续空白字符
WHITESPACE_PAT = re.compile(r'\s+')
def html_to_text(html) -> str:
    """Converts HTML content to plain text..
    1. 解码字节流，容错编码错误
    2. 移除 script/style 标签及内容
    3. 替换 img 为 [IMAGE: alt文本]
    4. 替换 a 标签为 [LINK: 文本 (链接)]
    5. 移除剩余所有 HTML 标签
    6. 清理空白字符，规范化文本
    Args:
        html (bytes): HTML content as bytes.
    Returns:
        str: Plain text extracted from HTML.
    """
    if not html:
        return ""
    
    try:
        html_str = html.decode('utf-8')
    except UnicodeDecodeError:
        print('Error when decode html bytes(utf-8), return none instead')
        return ""
    
    html_str = SCRIPT_STYLE_PAT.sub('', html_str)
    html_str = IMG_ALT_PAT.sub(r' [IMAGE: \1] ', html_str)
    html_str = re.sub(r'<img[^>]*>', ' [IMAGE] ', html_str, flags=re.IGNORECASE)
    
    def replace_a_tag(match):
        href = match.group(1)
        text = TAG_PAT.sub('', match.group(2)).strip()  # 移除 a 标签内的嵌套标签
        return f' [LINK: {text} ({href})] '
    html_str = A_TAG_PAT.sub(replace_a_tag, html_str)
    
    html_str = TAG_PAT.sub('', html_str)
    
    html_str = WHITESPACE_PAT.sub(' ', html_str).strip()
    
    return html_str
    
    
    

ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
phone_pattern = r'\+1[-\s]?(\(\d{3}\)|\d{3})[-\s]?\d{3}[-\s]?\d{4}'
def replace_pii(text: str) -> str:
    """Masks personally identifiable information (PII) from text with the specified masking formats.
    
    Mask clear PII patterns before further filtering. 
    
    - Replace U.S. Social Security numbers of the form XXX-XX-XXXX by converting every digit to X, 
    - Replace any 10-digit phone number prefixed with +1 by an all-X version (preserving the leading +). Leave other text untouched.
    Args:
        text (str): Candidate text.
    Returns:
        str: Text with PII obfuscated.
    """
    # Replace US social security numbers (XXX-XX-XXXX format)
    text = re.sub(ssn_pattern, "XXX-XX-XXXX", text)
    text = re.sub(phone_pattern, lambda match: '+1' + 'X' * (len(match.group(0)) - 2), text)
    return text
    
    
def _have_long_alphanumeric_sequence(paragraph: str) -> bool:
    '''检查段落是否包含超过 100 个没有空格的字母数字字符'''
    return bool(re.search(r'[A-Za-z0-9]{100,}', paragraph))

def _have_punctuation(paragraph: str) -> bool:
    '''检查有没有标点符号'''
    return bool(re.search(r'[!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~]', paragraph))
    

def clean_text(text: str) -> str:
    """Removes substrings identified as low-quality according to alphanumeric, whitespace and valid document checks.
    - Split the document into paragraphs with text.split("\n"), 
    - Drop paragraphs that contain more than 100 alphanumeric characters with no whitespace between them.
    - Drop paragraphs that do not contain punctuation. 
    - Join the surviving paragraphs with newline characters in their original order.
    Args:
        text (str): document to process.
    Returns:
        str: cleaned document
    """

    if text is None:
        return ''

    paragraphs = text.split("\n")
    
    cleaned_paragraphs = [
        p for p in paragraphs
        if not _have_long_alphanumeric_sequence(p) and _have_punctuation(p)
    ]
    
    return "\n".join(cleaned_paragraphs)


bad_words = None
def _init_bad_words():
    global bad_words
    with open('bad_word_list.txt', 'r', encoding='utf-8') as file:
        # 这里假设了内存能放下 bad_words
        bad_words = set(word.strip().lower() for word in file.readlines())


def heuristic_quality_filter(text: str) -> bool:
    """Rejects documents based on the presence of bad words and punctuation.
    Args:
        text (str): document to check
    Returns:
        bool: returns True if the document passes the filters, False otherwise.
             Specifically, ensure the text contains no entries from bad_word_list.txt, includes at least one character from string.punctuation
             
    """
    if bad_words is None:
        _init_bad_words()

    assert isinstance(bad_words, set)
    # 检查文档是否包含不良词汇
    text_lower = text.lower()
    if any(bad_word in text_lower for bad_word in bad_words):
        return False
    
    # 检查文档是否包含标点符号
    if any(punctuation in text for punctuation in string.punctuation):
        return True
    
    return False
    


def is_english_text(text: str, threshold = 0.5) -> bool:
    """Detects if text is primarily in English based on character distribution.
    Args:
        text (str): Text to analyze
    Returns:
        bool: True if text is primarily English, False otherwise
    """
    letters_only = re.sub(r'[^a-zA-Z]', '', text)
    letter_percentage = len(letters_only) / len(text) if text else 0
    
    return letter_percentage >= threshold

def _jaccard_similarity(set1: set, set2: set) -> float:
    '''Compute jaccard similarity'''
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def deduplicate_texts(texts: list[str], threshold: float = 0.9) -> list[str]:
    """通过 Jaccard 相似度去除重复句子。
    
    参数:
        texts (list[str]): 要去重的文本列表
        threshold (float): Jaccard 相似度阈值，超过该值的句子认为是重复的
    
    返回:
        list[str]: 去重后的文本列表
    """
    deduplicated_texts = []
    for text in texts:
        words = set(re.findall(r'\w+', text.lower()))
        if not any(_jaccard_similarity(words, set(re.findall(r'\w+', existing_text.lower()))) > threshold for existing_text in deduplicated_texts):
            deduplicated_texts.append(text)
    
    return deduplicated_texts
            


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type = str,  default = '', help = 'Specify the path for your warc file.')
    parser.add_argument('--dfname', type = str,  default = '', help = 'Specify the path where you stored topic_dataset.json')
    parser.add_argument('--num_records', type = int,  default=30, help = 'Specify the number of records you want to parse (only used for debugging with smaller sets)')
    parser.add_argument('--output', type = str,  default='cleaned_documents.txt', help = 'Output file for cleaned text documents')
    # parser.add_argument('--wet_name', type = str, default = '', help = 'Specify the path for your wet file.')
    args = parser.parse_args()

    if args.fname:
        seen = 0
        passes = 0

        with open(args.output, 'w', encoding='utf-8') as output_file:
            for url, html_text in read_warc_file(args.fname, args.num_records):
                seen += 1
                # print("Before HTML to text: ", str(html_text))
                text = html_to_text(html_text)
                # print("\n\n\nAfter HTML to text: ", text)
                cleaned_text = clean_text(text)
                # print("After cleaning: ", cleaned_text)
                cleaned_nopii_text = replace_pii(cleaned_text)
                # print("After PII removal: ", cleaned_nopii_text)
                passes_check = heuristic_quality_filter(cleaned_nopii_text)
                is_english = is_english_text(cleaned_nopii_text)
                print(url)
                print("Passes heuristic quality filter:", passes_check)
                print("Is English text:", is_english)
                if passes_check and is_english:
                    passes += 1
                    # Replace newlines with spaces to keep each document on one line
                    single_line_text = cleaned_nopii_text.replace('\n', ' ').replace('\r', ' ').strip()
                    output_file.write(single_line_text + '\n')
                    print("Saved cleaned English document to output file")
                elif passes_check and not is_english:
                    print("Document filtered out: not English")

        print(f"{passes} passed out of {seen} records processed.")
        print(f"Cleaned documents saved to: {args.output}")

    if args.dfname:
        with open(args.dfname, 'r') as f:
            raw_texts = json.load(f)
        raw_texts = [item['text'] for item in raw_texts['data']]
        deduplicated_texts = deduplicate_texts(raw_texts)
        print(f"{len(deduplicated_texts)} deduplicated out of {len(raw_texts)} records processed.")
    else:
        print("Usage: python homework.py --fname data.warc")
import json
from typing import Any, Dict, List

from llama_index.core.node_parser import SentenceSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 20):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def has_pic_tag(self, text: str) -> bool:
        return "<PIC>" in text

    def process_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    doc_id, text, image_ids = data

                    if len(text) <= self.splitter.chunk_size:
                        docs.append(
                            {
                                "doc_id": doc_id,
                                "text": text,
                                "image_ids": image_ids
                                if self.has_pic_tag(text)
                                else [],
                            }
                        )
                    else:
                        for chunk in self.splitter.split_text(text):
                            docs.append(
                                {
                                    "doc_id": doc_id,
                                    "text": chunk,
                                    "image_ids": image_ids
                                    if self.has_pic_tag(chunk)
                                    else [],
                                }
                            )

                except Exception as e:
                    print(f"[DocumentProcessor] 에러 (라인 {line_num}): {e}")
                    continue
        print(f"[DocumentProcessor] 총 {len(docs)}개 청크 생성")
        return docs

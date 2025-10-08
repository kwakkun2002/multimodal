import json
from typing import List

from ragas_eval.models.test_sample import Mqa


def load_mqa(jsonl_path: str, limit: int = None) -> List[Mqa]:
    samples = []

    # JSONL 파일을 한 줄씩 읽기
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            # JSON 파싱
            data = json.loads(line)
            # TestSample 객체로 변환하여 추가
            samples.append(
                Mqa(
                    id=data.get("id", ""),
                    question=data.get("question", ""),
                    ground_truth=data.get("ground_truth", ""),
                    images_list=data.get("images_list", []),
                )
            )

    print("[load_test_data] 로드된 샘플 개수:", len(samples))
    print("[load_test_data] 로드된 샘플 첫 번째 샘플:")
    print(samples[0])

    # 샘플 개수 제한이 있으면 적용
    if limit:
        print(f"[load_test_data] 샘플 개수 제한: {limit}")
        samples = samples[:limit]
    return samples

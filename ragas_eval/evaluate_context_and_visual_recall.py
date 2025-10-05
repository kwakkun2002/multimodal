# %%
# Jupyter notebook에서 실행 시 프로젝트 루트 경로를 Python 경로에 추가
import os
import sys
from pathlib import Path

import pandas as pd

# 현재 작업 디렉토리에서 프로젝트 루트를 찾아서 Python 경로에 추가
# notebook을 ragas_eval 디렉토리에서 실행하는 경우를 고려
current_dir = Path.cwd()
# ragas_eval 디렉토리에서 실행 중이면 부모 디렉토리(프로젝트 루트)로 이동
if current_dir.name == "ragas_eval":
    project_root = current_dir.parent
else:
    project_root = current_dir

# 프로젝트 루트를 Python 경로에 추가
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from ragas_eval.context_recall_evaluator import ContextRecallEvaluator
from ragas_eval.visual_recall_evaluator import VisualRecallEvaluator

# %%

# 평가 데이터 경로를 환경 변수로부터 읽기(기본값 제공)
jsonl_path = os.environ.get(
    "WIT_MQA_JSONL",
    # 기본 경로 설정
    "/home/kun/Desktop/multimodal/data/MRAMG-Bench/wit_mqa.jsonl",
)

# %%

# Context Recall 평가기 인스턴스 생성
context_evaluator = ContextRecallEvaluator()
# 평균 점수 및 기본 상세 결과, 보조 데이터 계산
avg_recall, details, samples, per_sample_retrieved_image_ids = (
    context_evaluator.evaluate_context_recall_at_k(jsonl_path=jsonl_path, limit=10)
)

# %%

# avg_recall을 DataFrame으로 변환 (단일 값이므로 1행 1열 DataFrame)
avg_recall_df = pd.DataFrame({"평균_Context_Recall": [avg_recall]}, index=["값"])

# details를 DataFrame으로 변환 (딕셔너리를 1행 DataFrame으로)
details_df = pd.DataFrame([details])

# samples를 DataFrame으로 변환 (리스트[딕셔너리] 형태)
# retrieved_contexts는 리스트이므로 길이로 표시하고, 전체 내용은 별도 열에 저장
samples_df = pd.DataFrame(samples)
# retrieved_contexts의 개수 정보 추가
samples_df["retrieved_contexts_count"] = samples_df["retrieved_contexts"].apply(len)

# per_sample_retrieved_image_ids를 DataFrame으로 변환
# 각 샘플의 이미지 ID 리스트를 별도 행으로 저장
per_sample_image_ids_df = pd.DataFrame(
    {
        "sample_index": range(len(per_sample_retrieved_image_ids)),
        "image_ids": per_sample_retrieved_image_ids,
        "image_ids_count": [len(ids) for ids in per_sample_retrieved_image_ids],
    }
)

# 각 DataFrame 출력
avg_recall_df
# %%

details_df
# %%

samples_df.head()
# %%
per_sample_image_ids_df.head()

# %%

# Visual Recall 평가기 인스턴스 생성
visual_evaluator = VisualRecallEvaluator()
# Visual Recall@K 점수를 상세 결과에 추가
visual_evaluator.add_visual_recall_to_details(
    details, samples, per_sample_retrieved_image_ids
)

# %%
# Visual Recall이 추가된 details를 DataFrame으로 변환
visual_details_df = pd.DataFrame([details])

# Visual Recall이 추가된 samples를 DataFrame으로 변환
visual_samples_df = pd.DataFrame(samples)
# retrieved_contexts의 개수 정보 추가
visual_samples_df["retrieved_contexts_count"] = visual_samples_df[
    "retrieved_contexts"
].apply(len)

# Visual Recall이 추가된 per_sample_retrieved_image_ids를 DataFrame으로 변환
visual_per_sample_image_ids_df = pd.DataFrame(
    {
        "sample_index": range(len(per_sample_retrieved_image_ids)),
        "image_ids": per_sample_retrieved_image_ids,
        "image_ids_count": [len(ids) for ids in per_sample_retrieved_image_ids],
    }
)

# 각 DataFrame 출력
visual_details_df
# %%
visual_samples_df.head()
# %%
visual_per_sample_image_ids_df.head()

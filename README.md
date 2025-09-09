## Multimodal RAG: CLIP + Milvus + MinIO + LLaVA

멀티모달 검색-생성(RAG) 데모 프로젝트입니다. 이미지를 `MinIO`에 저장하고 `CLIP` 임베딩을 `Milvus`에 저장한 뒤, 텍스트 질의로 가장 유사한 이미지를 찾고 `LLaVA`로 해당 이미지를 설명합니다. `LangChain` 검색기(`retriever`)와 간단한 파이프라인을 제공합니다.

- **벡터DB/오브젝트스토리지**: Milvus + MinIO (Docker Compose)
- **임베딩**: Hugging Face `openai/clip-vit-large-patch14`
- **생성 모델**: `llava-hf/llava-1.5-7b-hf` (기본 4bit 로딩)
- **파이프라인**: 텍스트→이미지 검색(CLIP+Milvus) → 이미지 설명(LLaVA)

### 빠른 시작(Quick Start)

#### 0) 요구 사항
- Docker, Docker Compose 설치
- Python 3.12, [`uv`](https://github.com/astral-sh/uv) 설치 (의존성/실행에 사용)
- NVIDIA GPU 권장(없어도 CPU로 동작하지만 느릴 수 있음)

#### 1) 백엔드 스택 기동(Milvus/MinIO/Attu)
```bash
docker compose up -d
```
- Milvus: `localhost:19530`
- MinIO API/Console: `localhost:9000` / `localhost:9001` (기본 계정: `minioadmin` / `minioadmin`)
- Attu(Milvus UI): `http://localhost:8000`

#### 2) Python 의존성 설치(uv)
```bash
uv python install 3.12
uv sync
```

#### 3) 데모 실행
```bash
uv run python demo_rag.py
```
- 기본 쿼리: "a cute cat"
- 검색 결과가 있다면, 최상위 이미지에 대해 LLaVA가 설명을 생성합니다.
- 샘플 이미지를 직접 넣어보고 싶다면 `demo_rag.py`의 `ingest` 플래그를 `True`로 바꾸세요.

### 프로젝트 구조(주요 파일)

- `MultiModalStore.py`: MinIO 업로드/프리사인 URL, CLIP 임베딩, Milvus 스키마/인덱스/검색을 캡슐화한 서비스 클래스
- `rag_langchain.py`: 
  - `CLIPMilvusRetriever`: `MultiModalStore`를 이용해 텍스트→이미지 검색 결과를 `LangChain Document`로 변환
  - `LlavaImageDescriber`: LLaVA 모델 로딩 및 이미지 설명 생성
  - `MultiModalRAG`: 검색기+생성기 결합 파이프라인
- `demo_rag.py`: 간단한 데모 엔트리포인트(검색→설명 출력)
- `docker-compose.yml`: Milvus, MinIO, Attu 구성
- `test_llava.py`: LLaVA 단독 테스트 스크립트(로컬 이미지 `kitten.png` 예시)

### 사용법 자세히

#### 데이터 적재(간단 예시)
`demo_rag.py`에는 PIL 이미지와 캡션을 넣어 한 번에 업로드+임베딩+Milvus 삽입하는 유틸이 포함되어 있습니다.

```python
from PIL import Image  # 로컬에서 예시 이미지를 만들거나 불러오기 위한 라이브러리
from demo_rag import ingest_images  # 예시 적재 함수(이미지/캡션 한번에 삽입)
from MultiModalStore import MinIOConfig, MilvusConfig, MultiModalStore  # 스토어 구성 요소들

minio_cfg = MinIOConfig()  # MinIO 연결 기본값 사용(로컬 docker-compose 기준)
milvus_cfg = MilvusConfig()  # Milvus 연결 기본값 사용(로컬 docker-compose 기준)
store = MultiModalStore(minio_cfg, milvus_cfg)  # 스토어 인스턴스 생성(업로드/임베딩/삽입/검색 담당)

img1 = Image.new("RGB", (256, 256), color=(255, 200, 200))  # 예시 이미지1(핑크 톤)
img2 = Image.new("RGB", (256, 256), color=(200, 200, 255))  # 예시 이미지2(블루 톤)
ingest_images(store, [img1, img2], ["핑크 사각형", "파란 사각형"])  # 두 이미지와 캡션을 삽입
```

#### 텍스트로 이미지 검색 + 설명 생성
`rag_langchain.py`의 구성요소를 직접 사용할 수도 있습니다.

```python
from rag_langchain import CLIPMilvusRetriever, LlavaImageDescriber, MultiModalRAG  # 검색기/생성기/파이프라인
from MultiModalStore import MinIOConfig, MilvusConfig, MultiModalStore  # 스토어 구성 요소

store = MultiModalStore(MinIOConfig(), MilvusConfig())  # MinIO/Milvus 연결 및 CLIP 로딩
retriever = CLIPMilvusRetriever(store=store, top_k=1)  # 텍스트 질의 상위 1개 검색
generator = LlavaImageDescriber()  # LLaVA 모델(기본 4bit 로드) 준비
rag = MultiModalRAG(retriever, generator)  # 검색+생성 파이프라인 구성

result = rag.invoke("a cute cat", use_top_k=1)  # 질의 수행 및 상위 이미지에 대한 설명 생성
print(result["answer"])  # 생성된 설명 출력
```

#### LLaVA 단독 테스트
```bash
uv run python test_llava.py
```

### 구성(설정)

`MultiModalStore.MinIOConfig`와 `MilvusConfig`는 기본적으로 로컬 Docker Compose 설정에 맞춰져 있습니다.

- MinIO: `endpoint=localhost:9000`, `access_key=secret_key=minioadmin`, `bucket=images`
- Milvus: `host=localhost`, `port=19530`, `collection_name=openclip_multimodal`, `metric_type=IP`, `index_type=HNSW`

필요 시 코드에서 다른 값으로 생성자에 전달해 변경하세요(현재 예제 코드는 환경변수 읽기 없이 코드 상에서 구성합니다).

### 트러블슈팅

- LLaVA 메모리 부족(OOM): 기본적으로 4bit 로딩을 사용하지만, GPU 메모리가 부족하면 더 작은 모델을 사용하거나 CPU로 실행하세요(속도 저하 주의).
- Milvus 접속 오류: `docker compose ps`로 서비스가 정상인지 확인하고, `19530` 포트가 열려있는지 점검하세요.
- MinIO 키 오류: 버킷/키가 맞는지 확인하고, presigned URL 생성 전에 객체가 존재하는지 검사합니다(코드에 존재 확인 로직 포함).

### 라이선스

별도 명시가 없는 파일은 MIT로 가정하거나, 필요 시 프로젝트에 맞게 갱신하세요.

### 참고

- 저장소: [`https://github.com/kwakkun2002/multimodal`](https://github.com/kwakkun2002/multimodal)



from io import BytesIO  # HTTP 응답 바이트를 메모리 버퍼로 감싸기 위한 BytesIO 임포트
from typing import List, Optional  # 타입 힌트를 위해 List와 Optional 임포트

import requests  # MinIO presigned URL로부터 이미지를 다운로드하기 위한 요청 라이브러리 임포트
import torch  # LLaVA 모델 가속 장치 선택 및 텐서 연산을 위한 PyTorch 임포트
from langchain_core.documents import (
    Document,  # LangChain 문서 객체(컨텍스트 단위) 임포트
)
from langchain_core.retrievers import (
    BaseRetriever,  # 커스텀 검색기를 만들기 위한 베이스 클래스 임포트
)
from minio.error import (
    S3Error,  # MinIO 객체 조회 시 오류 코드 확인을 위한 예외 클래스 임포트
)
from PIL import Image  # 이미지 로딩 및 처리를 위한 PIL 임포트
from transformers import (  # LLaVA 모델 및 프로세서를 로드하기 위한 Hugging Face Transformers 임포트
    AutoProcessor,  # 채팅 템플릿 적용과 입력 전처리를 담당하는 프로세서 클래스
    LlavaForConditionalGeneration,  # 이미지 조건 생성 모델인 LLaVA 모델 클래스
)

from MultiModalStore import (
    MultiModalStore,  # 기존 Milvus/MinIO/CLIP 래퍼 클래스를 재사용하기 위해 임포트
)


def _load_image_from_s3(
    minio_client, s3_uri: str
) -> (
    Image.Image
):  # MinIO 클라이언트와 s3 URI로부터 이미지를 로드하는 헬퍼 함수(미니오_이미지로드)
    if not s3_uri or not s3_uri.startswith("s3://"):  # s3 스킴 유효성 검사
        raise ValueError(f"유효하지 않은 s3 uri: {s3_uri}")  # 잘못된 경로라면 예외 발생
    path = s3_uri[len("s3://") :]  # 접두어 제거 후 버킷/키 추출 준비
    if "/" not in path:  # 버킷/키 구분자가 존재하는지 확인
        raise ValueError(
            f"유효하지 않은 s3 uri(키 누락): {s3_uri}"
        )  # 키 누락 시 예외 발생
    bucket, key = path.split("/", 1)  # '버킷', '키'로 분리
    try:
        obj = minio_client.get_object(bucket, key)  # 원본 키로 객체 조회 시도
    except S3Error as e:  # MinIO 오류 처리
        if e.code == "NoSuchKey" and not key.endswith(
            ".jpg"
        ):  # 키 없음이며 확장자 누락 가능성 판단
            alt_key = key + ".jpg"  # 대체 키 후보 생성(.jpg 추가)
            obj = minio_client.get_object(bucket, alt_key)  # 대체 키로 재시도
        else:
            raise  # 다른 오류는 상위로 전파
    try:
        data = obj.read()  # 바이트 데이터 전체 읽기
    finally:
        obj.close()  # 네트워크/파일 핸들 정리
    return Image.open(BytesIO(data)).convert(
        "RGB"
    )  # PIL 이미지로 열고 RGB로 일관화 후 반환


class CLIPMilvusRetriever(
    BaseRetriever
):  # MultiModalStore를 사용해 텍스트→이미지 검색을 수행하는 LangChain 검색기 클래스(클립_밀버스_리트리버)
    store: MultiModalStore  # 검색에 사용할 MultiModalStore 인스턴스(파이단틱 필드)
    top_k: int = 5  # 검색 결과 상위 k 설정(파이단틱 필드, 기본값 5)

    def _get_relevant_documents(
        self, query: str
    ) -> List[
        Document
    ]:  # LangChain이 호출하는 내부 검색 메서드 오버라이드(관련_문서_가져오기)
        hits = self.store.search_by_text(
            query, top_k=self.top_k
        )  # 텍스트 쿼리로 Milvus에서 상위 k개 유사 이미지 검색 실행
        docs: List[Document] = []  # LangChain Document 리스트를 생성할 준비
        for h in hits:  # 각 검색 결과에 대해 반복
            s3_uri: str = h.get("image_path")  # 결과의 MinIO 경로(s3 스킴)를 추출
            presigned: str = self.store.presigned_url(
                s3_uri
            )  # presigned URL을 생성해 HTTP로 접근 가능하게 변환
            content: str = (
                h.get("text") or ""
            )  # 캡션 텍스트를 컨텐츠로 사용(없으면 빈 문자열)
            meta = {  # 생성된 문서의 메타데이터 구성
                "milvus_id": h.get("id"),  # Milvus 기본키
                "distance": h.get("distance"),  # 유사도(내적: 클수록 유사)
                "image_path": s3_uri,  # MinIO 경로 보관
                "image_url": presigned,  # presigned URL 보관(다운로드에 활용)
            }
            docs.append(
                Document(page_content=content, metadata=meta)
            )  # 문서를 생성하여 결과 목록에 추가
        return docs  # LangChain이 사용할 문서 리스트 반환


class LlavaImageDescriber:  # LLaVA 모델을 사용해 이미지를 설명하는 생성기 클래스(라바_이미지_서술기)
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",  # 사용할 LLaVA 모델 식별자(허깅페이스 허브 경로)
        load_in_4bit: bool = True,  # 4비트 양자화 로딩으로 GPU 메모리 사용량 절감
    ):
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # 사용 가능한 장치를 선택(cuda 우선)
        dtype = (
            torch.bfloat16 if self.device == "cuda" else torch.float16
        )  # GPU면 bfloat16, 아니면 float16로 설정
        self.model = (
            LlavaForConditionalGeneration.from_pretrained(  # LLaVA 모델 가중치 로드
                pretrained_model_name_or_path=model_id,  # 모델 식별자 지정
                torch_dtype=dtype,  # 텐서 정밀도 지정
                low_cpu_mem_usage=True,  # CPU 메모리 사용량을 낮추는 옵션 활성화
                load_in_4bit=load_in_4bit,  # 4비트 로딩 활성화(가능한 경우)
            ).to(self.device)
        )  # 선택한 장치로 모델 이동
        self.processor = AutoProcessor.from_pretrained(
            model_id
        )  # 채팅 템플릿 및 이미지+텍스트 전처리기 로드

    def _build_prompt(
        self, user_query: str
    ) -> str:  # 사용자 쿼리를 기반으로 LLaVA용 채팅 프롬프트를 구성하는 내부 함수(프롬프트_구성)
        conversation = [  # LLaVA 채팅 대화 형식을 따른 입력 구성
            {
                "role": "user",  # 사용자 역할 지정
                "content": [  # 콘텐츠는 텍스트+이미지로 구성
                    {
                        "type": "text",  # 텍스트 타입 지정
                        "text": f"Explain the image in English, focusing on the query: {user_query}",  # 한국어 설명 요청과 쿼리 컨텍스트를 포함한 지시
                    },
                    {"type": "image"},  # 이미지 토큰 자리 표시
                ],
            }
        ]
        return self.processor.apply_chat_template(  # 프로세서의 템플릿 적용으로 모델 입력 텍스트 생성
            conversation,
            add_generation_prompt=True,  # 생성 프롬프트 추가 옵션 활성화
        )

    def describe(
        self, image: Image.Image, user_query: str, max_new_tokens: int = 200
    ) -> str:  # 단일 이미지를 쿼리 컨텍스트로 설명하는 메서드(설명)
        prompt = self._build_prompt(user_query)  # 사용자 쿼리로 프롬프트 생성
        inputs = self.processor(  # 이미지와 텍스트 프롬프트를 모델 입력으로 변환
            images=image,
            text=prompt,
            return_tensors="pt",  # 배치 차원을 가진 PyTorch 텐서 반환
        ).to(self.device, torch.float16)  # 장치 및 dtype 설정 후 이동
        output = self.model.generate(  # 생성 호출로 새 토큰을 생성
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 디코딩 길이 및 탐욕(decoding) 설정
        )
        text = self.processor.decode(  # 토큰 ID 시퀀스를 사람이 읽는 문자열로 복원
            output[0][2:],
            skip_special_tokens=True,  # 선행 특수 토큰 제거 및 특수 토큰 스킵
        )
        return text  # 생성된 한국어 설명 텍스트 반환

    @staticmethod
    def load_image_from_url(
        url: str, timeout: int = 30
    ) -> Image.Image:  # URL에서 이미지를 로드하는 유틸 함수(이미지_URL_로드)
        resp = requests.get(url, timeout=timeout)  # HTTP GET으로 이미지 바이트 다운로드
        resp.raise_for_status()  # 상태 코드 검증 및 예외 발생
        return Image.open(BytesIO(resp.content)).convert(
            "RGB"
        )  # 바이트를 PIL 이미지로 열고 RGB로 일관화


class MultiModalRAG:  # 검색기와 생성기를 결합한 간단한 RAG 파이프라인 클래스(멀티모달_RAG)
    def __init__(
        self, retriever: CLIPMilvusRetriever, generator: LlavaImageDescriber
    ):  # 의존성으로 검색기와 생성기를 주입하는 생성자
        self.retriever = retriever  # 검색기 보관
        self.generator = generator  # 생성기 보관

    def invoke(
        self, query: str, use_top_k: int = 1
    ) -> (
        dict
    ):  # 텍스트 쿼리로 검색 후 상위 이미지에 대해 설명을 생성하는 엔드포인트(호출)
        docs = self.retriever.get_relevant_documents(
            query
        )  # LangChain 방식으로 관련 문서(검색 결과) 가져오기
        if not docs:  # 검색 결과가 없을 때 처리
            return {
                "answer": "검색 결과가 없습니다.",  # 사용자에게 결과 없음 안내
                "contexts": [],  # 문맥 리스트 비움
            }

        top_docs = docs[:use_top_k]  # 상위 k개 결과만 사용(기본 1개)
        chosen = top_docs[0]  # 첫 번째 결과를 대표 이미지로 선택
        image_url: Optional[str] = chosen.metadata.get(
            "image_url"
        )  # presigned 이미지 URL 추출

        # MinIO에서 이미지 로드 (간결한 헬퍼 함수 사용)
        s3_uri: Optional[str] = chosen.metadata.get("image_path")  # s3 경로 가져오기
        img = _load_image_from_s3(
            self.retriever.store.minio, s3_uri
        )  # MinIO로부터 이미지 로드

        answer = self.generator.describe(
            img, query
        )  # LLaVA를 사용해 이미지 설명 생성(쿼리 컨텍스트 반영)

        return {
            "answer": answer,  # 생성된 최종 답변 텍스트
            "contexts": [
                d.model_dump() for d in top_docs
            ],  # 참고 컨텍스트(문서)들을 사전 형태로 반환(Pydantic v2 호환)
        }

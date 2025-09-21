import glob  # 파일 패턴 매칭을 위한 glob 모듈 임포트
import io  # 바이트 스트림 처리를 위한 io 모듈 임포트
import os  # 파일 시스템 경로 조작을 위한 os 모듈 임포트
from dataclasses import dataclass  # 데이터 클래스 정의를 위한 dataclass 임포트
from typing import Optional  # 타입 힌트를 위한 Optional 임포트

from minio import Minio  # MinIO 클라이언트 사용을 위한 임포트
from PIL import Image  # 이미지 처리를 위한 PIL 임포트
from tqdm import tqdm  # 진행 상황 표시를 위한 tqdm 임포트

# -------- 설정 클래스 --------


@dataclass
class MinIOConfig:  # MinIO 연결 설정을 담는 데이터 클래스
    """MinIO 연결과 버킷 정보를 담는 설정 클래스"""

    endpoint: str = (
        "localhost:9000"  # MinIO 서버 주소 (도커 컴포즈 내부면 "minio:9000")
    )
    access_key: str = "minioadmin"  # MinIO 액세스 키
    secret_key: str = "minioadmin"  # MinIO 시크릿 키
    secure: bool = False  # HTTP 사용 시 False, HTTPS 사용 시 True
    bucket: str = "wit-images"  # 사용할 버킷 이름 (MinIO 규칙에 맞는 소문자+하이픈)


class JPGUploader:  # JPG 파일들을 MinIO에 업로드하는 메인 클래스
    """JPG 이미지 파일들을 MinIO 버킷에 업로드하는 클래스"""

    def __init__(self, minio_cfg: MinIOConfig):  # MinIO 설정으로 초기화하는 생성자
        """MinIO 연결 설정과 클라이언트를 초기화"""
        self.minio_cfg = minio_cfg  # MinIO 설정 저장
        self.minio = None  # MinIO 클라이언트는 나중에 초기화
        self._setup_minio()  # MinIO 클라이언트 설정 메서드 호출

    def _setup_minio(self):  # MinIO 클라이언트 설정 메서드
        """MinIO 클라이언트를 생성하고 버킷을 확인/생성"""
        # MinIO 클라이언트 생성
        self.minio = Minio(
            self.minio_cfg.endpoint,
            access_key=self.minio_cfg.access_key,
            secret_key=self.minio_cfg.secret_key,
            secure=self.minio_cfg.secure,
        )

        # 버킷 존재 여부 확인 후 없으면 생성
        if not self.minio.bucket_exists(self.minio_cfg.bucket):
            self.minio.make_bucket(self.minio_cfg.bucket)
            print(f"버킷 '{self.minio_cfg.bucket}'이(가) 생성되었습니다.")
        else:
            print(f"버킷 '{self.minio_cfg.bucket}'이(가) 이미 존재합니다.")

    def upload_jpg_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        preserve_original_name: bool = True,
    ) -> str:  # 단일 JPG 파일 업로드 메서드
        """단일 JPG 파일을 MinIO에 업로드"""
        if object_name is None:  # 객체 이름이 지정되지 않은 경우
            if preserve_original_name:  # 원본 파일명 그대로 사용
                object_name = os.path.basename(file_path)  # 파일 이름 사용
            else:  # 고유한 이름 생성 (중복 방지)
                import uuid

                base_name = os.path.splitext(os.path.basename(file_path))[0]
                object_name = f"{base_name}_{uuid.uuid4().hex[:8]}.jpg"

        # PIL을 사용해 이미지 열기 (JPEG 형식 검증)
        try:
            image = Image.open(file_path)  # 이미지 파일 열기
            image = image.convert("RGB")  # RGB 모드로 변환 (알파 채널 제거)
        except Exception as e:  # 이미지 파일이 아닌 경우 예외 처리
            raise ValueError(f"'{file_path}'은(는) 유효한 이미지 파일이 아닙니다: {e}")

        # 바이트 스트림으로 변환
        buf = io.BytesIO()  # 메모리 버퍼 생성
        image.save(buf, format="JPEG", quality=95)  # JPEG로 인코딩하여 버퍼에 저장
        data = buf.getvalue()  # 버퍼에서 바이트 데이터 추출

        # MinIO에 업로드
        self.minio.put_object(
            self.minio_cfg.bucket,  # 대상 버킷
            object_name,  # 객체 이름
            data=io.BytesIO(data),  # 바이트 데이터
            length=len(data),  # 데이터 길이
            content_type="image/jpeg",  # MIME 타입
        )

        return (
            f"s3://{self.minio_cfg.bucket}/{object_name}"  # 업로드된 객체의 S3 URI 반환
        )

    def upload_directory(
        self,
        directory_path: str,
        pattern: str = "*.jpg",
        preserve_original_name: bool = True,
    ) -> list:  # 디렉토리 내 모든 JPG 파일 업로드 메서드
        """지정된 디렉토리에서 패턴과 일치하는 모든 파일을 업로드"""
        # 디렉토리에서 JPG 파일들 찾기
        jpg_files = glob.glob(
            os.path.join(directory_path, pattern)
        )  # 패턴과 일치하는 파일 경로들

        if not jpg_files:  # 찾은 파일이 없는 경우
            print(
                f"디렉토리 '{directory_path}'에서 '{pattern}' 패턴과 일치하는 파일을 찾을 수 없습니다."
            )
            return []

        print(f"총 {len(jpg_files)}개의 파일을 발견했습니다.")
        print(
            f"업로드할 파일들: {jpg_files[:3]}..."
            if len(jpg_files) > 3
            else f"업로드할 파일들: {jpg_files}"
        )

        uploaded_files = []  # 업로드 성공한 파일들의 정보를 저장할 리스트

        # tqdm을 사용해 진행률 표시하며 파일들을 업로드
        for file_path in tqdm(jpg_files, desc="파일 업로드 중", unit="개"):
            try:
                # 파일 업로드 실행
                s3_uri = self.upload_jpg_file(
                    file_path, preserve_original_name=preserve_original_name
                )
                uploaded_files.append(
                    {
                        "local_path": file_path,  # 로컬 파일 경로
                        "s3_uri": s3_uri,  # MinIO S3 URI
                        "object_name": os.path.basename(file_path),  # 객체 이름
                    }
                )
                print(f"✓ {os.path.basename(file_path)} 업로드 완료")
            except Exception as e:  # 업로드 실패 시
                print(f"✗ {os.path.basename(file_path)} 업로드 실패: {e}")

        return uploaded_files  # 업로드된 파일들의 정보 반환

    def get_upload_summary(
        self, uploaded_files: list
    ) -> dict:  # 업로드 결과 요약 메서드
        """업로드 결과를 요약해서 반환"""
        total_files = len(uploaded_files)  # 업로드된 파일 수
        return {
            "total_files": total_files,  # 총 업로드된 파일 수
            "uploaded_files": uploaded_files,  # 업로드된 파일들의 상세 정보
            "success_rate": f"{total_files / len(glob.glob(os.path.join('/home/kun/Desktop/multimodal/data/MRAMG-Bench/IMAGE/images/WIT', '*.jpg'))) * 100:.1f}%"
            if total_files > 0
            else "0%",
        }


def main():  # 메인 함수 - 스크립트 실행 시 호출
    """스크립트의 메인 실행 함수"""
    # MinIO 설정 생성 (기본값 사용)
    minio_cfg = MinIOConfig()

    # 업로드할 디렉토리 경로
    directory_path = "/home/kun/Desktop/multimodal/data/MRAMG-Bench/IMAGE/images/WIT"

    # JPGUploader 인스턴스 생성
    uploader = JPGUploader(minio_cfg)

    # 파일명 처리 방식 설정
    preserve_original_name = True  # True: 원본 파일명 그대로, False: 고유한 이름 생성

    # 디렉토리에서 모든 JPG 파일 업로드
    uploaded_files = uploader.upload_directory(
        directory_path, preserve_original_name=preserve_original_name
    )

    # 업로드 결과 요약
    summary = uploader.get_upload_summary(uploaded_files)

    print("\n" + "=" * 50)
    print("업로드 완료 요약")
    print("=" * 50)
    print(f"총 업로드된 파일 수: {summary['total_files']}")
    print(f"성공률: {summary['success_rate']}")
    print("=" * 50)

    # 업로드된 파일들의 S3 URI 출력
    if uploaded_files:
        print("업로드된 파일들의 S3 URI:")
        for file_info in uploaded_files[:5]:  # 처음 5개만 표시
            print(f"  - {file_info['s3_uri']}")
        if len(uploaded_files) > 5:
            print(f"  ... 그리고 {len(uploaded_files) - 5}개의 파일 더")


if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 main 함수 호출
    main()

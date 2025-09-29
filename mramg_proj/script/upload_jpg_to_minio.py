from JPGUploader import JPGUploader
from WitImagesStorageConfig import WitImagesStorageConfig




def main():  # 메인 함수 - 스크립트 실행 시 호출
    """스크립트의 메인 실행 함수"""
    # MinIO 설정 생성 (기본값 사용)
    minio_cfg = WitImagesStorageConfig()

    # 업로드할 디렉토리 경로
    directory_path = "/home/kun/Desktop/multimodal/data/MRAMG-Bench/IMAGE/images/WIT"

    # JPGUploader 인스턴스 생성
    uploader = JPGUploader(minio_cfg)

    # 디렉토리에서 모든 JPG 파일 업로드
    uploaded_files = uploader.upload_directory(
        directory_path,
    )

    # 업로드 결과 요약
    summary = uploader.get_upload_summary(uploaded_files)

    print("\n" + "=" * 50)
    print("업로드 완료 요약")
    print("=" * 50)
    print(f"총 업로드된 파일 수: {summary['total_files']}")
    print(f"성공률: {summary['success_rate']}")
    print("=" * 50)


if __name__ == "__main__":
    main()

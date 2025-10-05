# Docker 관리 도구 사용법

이 도구는 `docker-compose.yml` 파일을 기반으로 Milvus 벡터 데이터베이스와 관련 서비스들을 쉽게 관리할 수 있는 bash 스크립트입니다.

## 🚀 빠른 시작

### 기본 사용법
```bash
# 도움말 보기
./docker-manager.sh help

# 모든 서비스 시작
./docker-manager.sh start

# 모든 서비스 중지
./docker-manager.sh stop

# 서비스 상태 확인
./docker-manager.sh status
```

## 📋 지원하는 명령어

### 1. 서비스 관리
- **start** - 모든 서비스 또는 특정 서비스 시작
- **stop** - 모든 서비스 또는 특정 서비스 중지  
- **restart** - 모든 서비스 또는 특정 서비스 재시작

### 2. 모니터링
- **status** - 모든 서비스의 상태 확인
- **logs** - 서비스 로그 실시간 확인
- **health** - 서비스 헬스체크 상태 확인

### 3. 유지보수
- **build** - 서비스 이미지 빌드
- **clean** - 중지된 컨테이너와 사용하지 않는 이미지 정리
- **reset** - 모든 서비스 중지 후 볼륨까지 완전 삭제

## 🏗️ 서비스 구성

이 도구는 다음 서비스들을 관리합니다:

| 서비스 | 설명 | 포트 | 용도 |
|--------|------|------|------|
| **etcd** | 분산 키-값 저장소 | 2379 | Milvus 메타데이터 저장 |
| **minio** | 객체 스토리지 | 9000, 9001 | 벡터 데이터 저장 |
| **standalone** | Milvus 벡터 DB | 19530, 9091 | 벡터 검색 엔진 |
| **attu** | Milvus 관리 UI | 3000 | 웹 기반 관리 인터페이스 |

## 💡 사용 예시

### 전체 서비스 관리
```bash
# 모든 서비스 시작
./docker-manager.sh start

# 모든 서비스 상태 확인
./docker-manager.sh status

# 모든 서비스 로그 확인
./docker-manager.sh logs

# 모든 서비스 중지
./docker-manager.sh stop
```

### 개별 서비스 관리
```bash
# Milvus만 시작
./docker-manager.sh start standalone

# MinIO 로그 확인 (최근 100줄)
./docker-manager.sh logs minio 100

# etcd 서비스 재시작
./docker-manager.sh restart etcd
```

### 헬스체크 및 모니터링
```bash
# 모든 서비스 헬스체크 상태 확인
./docker-manager.sh health

# 특정 서비스 로그 실시간 모니터링
./docker-manager.sh logs standalone
```

### 유지보수 작업
```bash
# 사용하지 않는 Docker 리소스 정리
./docker-manager.sh clean

# 모든 서비스와 데이터 완전 삭제 (주의!)
./docker-manager.sh reset
```

## 🔧 고급 사용법

### 로그 확인 옵션
```bash
# 기본 (최근 50줄)
./docker-manager.sh logs

# 최근 100줄
./docker-manager.sh logs standalone 100

# 실시간 로그 모니터링
./docker-manager.sh logs minio
```

### 서비스 시작 순서
서비스들은 다음 순서로 자동 시작됩니다:
1. **etcd** - 메타데이터 저장소
2. **minio** - 객체 스토리지  
3. **standalone** - Milvus 벡터 데이터베이스
4. **attu** - 웹 관리 인터페이스

## ⚠️ 주의사항

### reset 명령어 사용 시
- `reset` 명령어는 **모든 데이터를 삭제**합니다
- 볼륨 데이터가 완전히 삭제되며 **복구할 수 없습니다**
- 신중하게 사용하세요

### 포트 충돌 확인
다음 포트들이 사용되므로 다른 서비스와 충돌하지 않는지 확인하세요:
- 9002, 9003 (MinIO)
- 19532, 9092 (Milvus)
- 8001 (Attu UI)

## 🐛 문제 해결

### Docker가 실행되지 않는 경우
```bash
# Docker 서비스 시작 (Ubuntu/Debian)
sudo systemctl start docker

# Docker 서비스 상태 확인
sudo systemctl status docker
```

### 권한 문제
```bash
# 스크립트 실행 권한 부여
chmod +x docker-manager.sh
```

### 포트 충돌
```bash
# 사용 중인 포트 확인
sudo netstat -tulpn | grep :19532
sudo netstat -tulpn | grep :9002
```

## 📊 서비스 접속 정보

서비스가 정상적으로 시작된 후 다음 URL로 접속할 수 있습니다:

- **Milvus 관리 UI (Attu)**: http://localhost:8001
- **MinIO 콘솔**: http://localhost:9003
  - 사용자명: `minioadmin`
  - 비밀번호: `minioadmin`

## 🔄 워크플로우 예시

### 개발 환경 설정
```bash
# 1. 모든 서비스 시작
./docker-manager.sh start

# 2. 서비스 상태 확인
./docker-manager.sh status

# 3. 헬스체크 확인
./docker-manager.sh health

# 4. Milvus UI 접속
# http://localhost:8001
```

### 개발 종료 시
```bash
# 모든 서비스 중지
./docker-manager.sh stop
```

### 완전 초기화 (필요시)
```bash
# 모든 데이터 삭제 후 재시작
./docker-manager.sh reset
./docker-manager.sh start
```

이 도구를 사용하면 Milvus 벡터 데이터베이스 환경을 쉽고 안전하게 관리할 수 있습니다! 🎉

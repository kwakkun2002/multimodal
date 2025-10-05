#!/bin/bash

# Docker Compose 기반 서비스 관리 도구
# Milvus 벡터 데이터베이스와 관련 서비스들을 쉽게 관리할 수 있는 스크립트

# 색상 코드 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Docker Compose 파일 경로
COMPOSE_FILE="docker-compose.yml"

# 도움말 메시지 출력 함수
show_help() {
    echo -e "${CYAN}Docker Compose 관리 도구${NC}"
    echo -e "${CYAN}========================${NC}"
    echo ""
    echo -e "${YELLOW}사용법:${NC} $0 [명령어] [서비스명]"
    echo ""
    echo -e "${YELLOW}명령어:${NC}"
    echo -e "  ${GREEN}start${NC}     - 모든 서비스 시작"
    echo -e "  ${GREEN}stop${NC}      - 모든 서비스 중지"
    echo -e "  ${GREEN}restart${NC}   - 모든 서비스 재시작"
    echo -e "  ${GREEN}status${NC}    - 서비스 상태 확인"
    echo -e "  ${GREEN}logs${NC}      - 서비스 로그 확인"
    echo -e "  ${GREEN}build${NC}     - 서비스 이미지 빌드"
    echo -e "  ${GREEN}clean${NC}     - 중지된 컨테이너와 사용하지 않는 이미지 정리"
    echo -e "  ${GREEN}reset${NC}     - 모든 서비스 중지 후 볼륨까지 삭제"
    echo -e "  ${GREEN}health${NC}    - 서비스 헬스체크 상태 확인"
    echo -e "  ${GREEN}help${NC}      - 이 도움말 표시"
    echo ""
    echo -e "${YELLOW}서비스명:${NC}"
    echo -e "  ${BLUE}etcd${NC}       - etcd 서비스"
    echo -e "  ${BLUE}minio${NC}      - MinIO 객체 스토리지"
    echo -e "  ${BLUE}standalone${NC} - Milvus 벡터 데이터베이스"
    echo -e "  ${BLUE}attu${NC}       - Milvus 관리 UI"
    echo ""
    echo -e "${YELLOW}예시:${NC}"
    echo -e "  $0 start                    # 모든 서비스 시작"
    echo -e "  $0 start standalone         # Milvus만 시작"
    echo -e "  $0 logs minio               # MinIO 로그 확인"
    echo -e "  $0 status                   # 모든 서비스 상태 확인"
}

# 에러 메시지 출력 함수
print_error() {
    echo -e "${RED}❌ 오류: $1${NC}" >&2
}

# 성공 메시지 출력 함수
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 정보 메시지 출력 함수
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 경고 메시지 출력 함수
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Docker Compose 파일 존재 확인
check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "Docker Compose 파일을 찾을 수 없습니다: $COMPOSE_FILE"
        exit 1
    fi
}

# Docker가 실행 중인지 확인
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker가 실행되지 않았거나 접근할 수 없습니다."
        print_info "Docker를 시작한 후 다시 시도해주세요."
        exit 1
    fi
}

# 모든 서비스 시작
start_services() {
    local service_name="$1"
    
    print_info "서비스를 시작하는 중..."
    
    if [ -n "$service_name" ]; then
        # 특정 서비스만 시작
        print_info "$service_name 서비스를 시작합니다..."
        if docker-compose -f "$COMPOSE_FILE" up -d "$service_name"; then
            print_success "$service_name 서비스가 성공적으로 시작되었습니다."
        else
            print_error "$service_name 서비스 시작에 실패했습니다."
            exit 1
        fi
    else
        # 모든 서비스 시작
        print_info "모든 서비스를 시작합니다..."
        if docker-compose -f "$COMPOSE_FILE" up -d; then
            print_success "모든 서비스가 성공적으로 시작되었습니다."
            print_info "서비스 시작 순서: etcd → minio → standalone → attu"
        else
            print_error "서비스 시작에 실패했습니다."
            exit 1
        fi
    fi
}

# 모든 서비스 중지
stop_services() {
    local service_name="$1"
    
    print_info "서비스를 중지하는 중..."
    
    if [ -n "$service_name" ]; then
        # 특정 서비스만 중지
        print_info "$service_name 서비스를 중지합니다..."
        if docker-compose -f "$COMPOSE_FILE" stop "$service_name"; then
            print_success "$service_name 서비스가 성공적으로 중지되었습니다."
        else
            print_error "$service_name 서비스 중지에 실패했습니다."
            exit 1
        fi
    else
        # 모든 서비스 중지
        print_info "모든 서비스를 중지합니다..."
        if docker-compose -f "$COMPOSE_FILE" stop; then
            print_success "모든 서비스가 성공적으로 중지되었습니다."
        else
            print_error "서비스 중지에 실패했습니다."
            exit 1
        fi
    fi
}

# 모든 서비스 재시작
restart_services() {
    local service_name="$1"
    
    print_info "서비스를 재시작하는 중..."
    
    if [ -n "$service_name" ]; then
        # 특정 서비스만 재시작
        print_info "$service_name 서비스를 재시작합니다..."
        if docker-compose -f "$COMPOSE_FILE" restart "$service_name"; then
            print_success "$service_name 서비스가 성공적으로 재시작되었습니다."
        else
            print_error "$service_name 서비스 재시작에 실패했습니다."
            exit 1
        fi
    else
        # 모든 서비스 재시작
        print_info "모든 서비스를 재시작합니다..."
        if docker-compose -f "$COMPOSE_FILE" restart; then
            print_success "모든 서비스가 성공적으로 재시작되었습니다."
        else
            print_error "서비스 재시작에 실패했습니다."
            exit 1
        fi
    fi
}

# 서비스 상태 확인
show_status() {
    print_info "서비스 상태를 확인하는 중..."
    echo ""
    
    # Docker Compose 서비스 상태 표시
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    print_info "서비스별 상세 정보:"
    echo ""
    
    # 각 서비스의 상태를 개별적으로 확인
    services=("etcd" "minio" "standalone" "attu")
    
    for service in "${services[@]}"; do
        echo -e "${CYAN}=== $service ===${NC}"
        
        # 컨테이너 상태 확인
        container_status=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null)
        if [ -n "$container_status" ]; then
            # 컨테이너가 존재하는 경우
            container_name=$(docker inspect --format='{{.Name}}' "$container_status" 2>/dev/null | sed 's/\///')
            container_state=$(docker inspect --format='{{.State.Status}}' "$container_status" 2>/dev/null)
            container_health=$(docker inspect --format='{{.State.Health.Status}}' "$container_status" 2>/dev/null)
            
            echo -e "컨테이너: ${GREEN}$container_name${NC}"
            echo -e "상태: ${GREEN}$container_state${NC}"
            
            if [ "$container_health" != "<no value>" ] && [ "$container_health" != "" ]; then
                if [ "$container_health" = "healthy" ]; then
                    echo -e "헬스체크: ${GREEN}$container_health${NC}"
                else
                    echo -e "헬스체크: ${YELLOW}$container_health${NC}"
                fi
            fi
            
            # 포트 정보 표시
            ports=$(docker inspect --format='{{range $p, $conf := .NetworkSettings.Ports}}{{if $conf}}{{$p}} {{end}}{{end}}' "$container_status" 2>/dev/null)
            if [ -n "$ports" ]; then
                echo -e "포트: ${BLUE}$ports${NC}"
            fi
        else
            echo -e "상태: ${RED}중지됨${NC}"
        fi
        
        echo ""
    done
}

# 서비스 로그 확인
show_logs() {
    local service_name="$1"
    local lines="${2:-50}"
    
    if [ -n "$service_name" ]; then
        print_info "$service_name 서비스의 로그를 확인합니다 (최근 $lines 줄)..."
        echo ""
        docker-compose -f "$COMPOSE_FILE" logs --tail="$lines" -f "$service_name"
    else
        print_info "모든 서비스의 로그를 확인합니다 (최근 $lines 줄)..."
        echo ""
        docker-compose -f "$COMPOSE_FILE" logs --tail="$lines" -f
    fi
}

# 서비스 이미지 빌드
build_services() {
    local service_name="$1"
    
    print_info "서비스 이미지를 빌드하는 중..."
    
    if [ -n "$service_name" ]; then
        # 특정 서비스만 빌드
        print_info "$service_name 서비스 이미지를 빌드합니다..."
        if docker-compose -f "$COMPOSE_FILE" build "$service_name"; then
            print_success "$service_name 서비스 이미지가 성공적으로 빌드되었습니다."
        else
            print_error "$service_name 서비스 이미지 빌드에 실패했습니다."
            exit 1
        fi
    else
        # 모든 서비스 빌드
        print_info "모든 서비스 이미지를 빌드합니다..."
        if docker-compose -f "$COMPOSE_FILE" build; then
            print_success "모든 서비스 이미지가 성공적으로 빌드되었습니다."
        else
            print_error "서비스 이미지 빌드에 실패했습니다."
            exit 1
        fi
    fi
}

# 정리 작업
clean_up() {
    print_warning "중지된 컨테이너와 사용하지 않는 이미지를 정리합니다..."
    
    # 중지된 컨테이너 제거
    print_info "중지된 컨테이너를 제거하는 중..."
    docker container prune -f
    
    # 사용하지 않는 이미지 제거
    print_info "사용하지 않는 이미지를 제거하는 중..."
    docker image prune -f
    
    # 사용하지 않는 네트워크 제거
    print_info "사용하지 않는 네트워크를 제거하는 중..."
    docker network prune -f
    
    print_success "정리 작업이 완료되었습니다."
}

# 완전 초기화 (볼륨까지 삭제)
reset_all() {
    print_warning "⚠️  경고: 이 작업은 모든 데이터를 삭제합니다!"
    print_warning "볼륨 데이터가 모두 삭제되며 복구할 수 없습니다."
    echo ""
    
    read -p "정말로 모든 서비스와 데이터를 삭제하시겠습니까? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ] || [ "$confirm" = "y" ]; then
        print_info "모든 서비스를 중지하고 볼륨을 삭제하는 중..."
        
        # 모든 서비스 중지 및 제거
        docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans
        
        # 볼륨 디렉토리 삭제
        if [ -d "volumes" ]; then
            print_info "볼륨 디렉토리를 삭제하는 중..."
            rm -rf volumes
        fi
        
        print_success "모든 서비스와 데이터가 삭제되었습니다."
    else
        print_info "작업이 취소되었습니다."
    fi
}

# 헬스체크 상태 확인
check_health() {
    print_info "서비스 헬스체크 상태를 확인하는 중..."
    echo ""
    
    services=("etcd" "minio" "standalone" "attu")
    
    for service in "${services[@]}"; do
        echo -e "${CYAN}=== $service 헬스체크 ===${NC}"
        
        container_id=$(docker-compose -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null)
        if [ -n "$container_id" ]; then
            # 컨테이너가 실행 중인 경우
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_id" 2>/dev/null)
            
            if [ "$health_status" = "healthy" ]; then
                echo -e "상태: ${GREEN}✅ Healthy${NC}"
            elif [ "$health_status" = "unhealthy" ]; then
                echo -e "상태: ${RED}❌ Unhealthy${NC}"
            elif [ "$health_status" = "starting" ]; then
                echo -e "상태: ${YELLOW}🔄 Starting${NC}"
            else
                echo -e "상태: ${BLUE}ℹ️  No health check${NC}"
            fi
            
            # 헬스체크 로그 표시
            if [ "$health_status" != "<no value>" ] && [ "$health_status" != "" ]; then
                echo "헬스체크 로그:"
                docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' "$container_id" 2>/dev/null | tail -3
            fi
        else
            echo -e "상태: ${RED}❌ 컨테이너가 실행되지 않음${NC}"
        fi
        
        echo ""
    done
}

# 메인 함수
main() {
    # Docker Compose 파일과 Docker 상태 확인
    check_compose_file
    check_docker
    
    # 명령어 파싱
    case "${1:-help}" in
        "start")
            start_services "$2"
            ;;
        "stop")
            stop_services "$2"
            ;;
        "restart")
            restart_services "$2"
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$2" "$3"
            ;;
        "build")
            build_services "$2"
            ;;
        "clean")
            clean_up
            ;;
        "reset")
            reset_all
            ;;
        "health")
            check_health
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "알 수 없는 명령어: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"

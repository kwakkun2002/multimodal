"""
Context Recall Evaluator 클래스의 각 함수 작동을 시각적으로 확인하는 테스트 코드

이 모듈은 실제 단위 테스트보다는 각 함수의 입력과 출력, 처리 과정을
시각적으로 확인하여 클래스의 동작 원리를 이해하는 것을 목표로 합니다.
"""

import os
from typing import Dict, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from ragas_eval.context_recall_evaluator import ContextRecallEvaluator, load_test_data

# 콘솔 출력을 위한 Rich Console 객체 생성
console = Console()

# Typer 앱 생성
app = typer.Typer(
    name="visualize-context-recall-test",
    help="Context Recall Evaluator 함수 시각화 테스트 도구",
    add_completion=False,
)


def visualize_load_test_data_function(jsonl_path: str):
    """
    load_test_data 함수의 동작을 시각화합니다.

    JSONL 파일에서 데이터를 읽어 파싱하는 과정을 보여줍니다.
    """
    # 섹션 제목 출력
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print(
        "📂 [bold cyan]함수 테스트: load_test_data()[/bold cyan]", style="bold"
    )
    console.print("=" * 80 + "\n", style="bold cyan")

    # 입력 파일 경로 표시
    console.print(f"[yellow]입력 파일 경로:[/yellow] {jsonl_path}")

    # 파일 존재 여부 확인
    if not os.path.exists(jsonl_path):
        console.print(f"[red]❌ 파일이 존재하지 않습니다: {jsonl_path}[/red]")
        return None

    try:
        # 실제 함수 실행
        samples = load_test_data(jsonl_path)

        # 결과 통계 출력
        console.print(f"\n[green]✓ 로드된 샘플 개수:[/green] {len(samples)}개\n")

        # 각 샘플을 테이블로 시각화
        for index, sample in enumerate(samples[:3]):  # 처음 3개만 표시
            table = Table(
                title=f"샘플 #{index + 1}",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("필드", style="cyan", width=20)
            table.add_column("값", style="green")

            # 각 필드를 테이블 행으로 추가
            table.add_row("ID", str(sample.get("id", "N/A")))
            table.add_row("질문", sample.get("question", "N/A")[:100] + "...")
            table.add_row("정답", sample.get("ground_truth", "N/A")[:100] + "...")
            table.add_row("이미지 리스트", str(sample.get("images_list", [])[:5]))

            console.print(table)
            console.print()

        # 전체 샘플 수가 3개보다 많으면 안내 메시지 출력
        if len(samples) > 3:
            console.print(
                f"[dim]... 외 {len(samples) - 3}개의 샘플이 더 있습니다.[/dim]\n"
            )

        return samples

    except Exception as error:
        # 에러 발생 시 에러 메시지 출력
        console.print(f"[red]❌ 에러 발생:[/red] {str(error)}")
        return None


def visualize_build_evaluator_llm_function(evaluator: ContextRecallEvaluator):
    """
    _build_evaluator_llm 함수의 동작을 시각화합니다.

    OpenAI LLM을 생성하는 과정과 설정을 보여줍니다.
    """
    # 섹션 제목 출력
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print(
        "🤖 [bold cyan]함수 테스트: _build_evaluator_llm()[/bold cyan]", style="bold"
    )
    console.print("=" * 80 + "\n", style="bold cyan")

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        console.print(
            f"[green]✓ OPENAI_API_KEY 확인됨:[/green] {api_key[:10]}...{api_key[-4:]}"
        )
    else:
        console.print("[red]❌ OPENAI_API_KEY가 설정되지 않았습니다.[/red]")
        return None

    try:
        # 모델명 지정
        model_name = "gpt-4o"
        console.print(f"[yellow]사용할 모델:[/yellow] {model_name}\n")

        # LLM 빌드 실행
        evaluator_llm = evaluator._build_evaluator_llm(model_name)

        # LLM 객체 정보 출력
        console.print("[green]✓ LLM 빌더 생성 완료[/green]")
        console.print(f"[dim]LLM 타입:[/dim] {type(evaluator_llm).__name__}")
        console.print(
            f"[dim]LLM 클래스:[/dim] {evaluator_llm.__class__.__module__}.{evaluator_llm.__class__.__name__}\n"
        )

        return evaluator_llm

    except Exception as error:
        # 에러 발생 시 에러 메시지 출력
        console.print(f"[red]❌ 에러 발생:[/red] {str(error)}")
        return None


def visualize_search_contexts_function(
    evaluator: ContextRecallEvaluator, query: str, top_k: int = 5
):
    """
    _search_contexts 함수의 동작을 시각화합니다.

    벡터 스토어에서 쿼리와 유사한 문서를 검색하는 과정을 보여줍니다.
    """
    # 섹션 제목 출력
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print(
        "🔍 [bold cyan]함수 테스트: _search_contexts()[/bold cyan]", style="bold"
    )
    console.print("=" * 80 + "\n", style="bold cyan")

    # 입력 파라미터 출력
    console.print(f"[yellow]검색 쿼리:[/yellow] {query}")
    console.print(f"[yellow]Top K:[/yellow] {top_k}\n")

    try:
        # 검색 실행
        search_results = evaluator._search_contexts(query, top_k)

        # 검색 결과 통계
        console.print(f"[green]✓ 검색 완료:[/green] {len(search_results)}개의 결과\n")

        # 검색 결과를 트리 구조로 시각화
        tree = Tree("🔎 검색 결과", guide_style="bold bright_blue")

        for index, result in enumerate(search_results[:5]):  # 처음 5개만 표시
            # 각 결과를 트리 노드로 추가
            result_node = tree.add(f"[cyan]결과 #{index + 1}[/cyan]")

            # 텍스트 내용 추가
            text_content = result.get("text", "N/A")
            if len(text_content) > 100:
                text_content = text_content[:100] + "..."
            result_node.add(f"[green]텍스트:[/green] {text_content}")

            # 점수가 있으면 추가
            if "score" in result:
                result_node.add(f"[yellow]유사도 점수:[/yellow] {result['score']:.4f}")

            # 이미지 ID가 있으면 추가
            if "image_ids" in result and result["image_ids"]:
                result_node.add(
                    f"[magenta]이미지 ID:[/magenta] {result['image_ids'][:3]}"
                )

        console.print(tree)
        console.print()

        # 전체 결과 수가 5개보다 많으면 안내 메시지
        if len(search_results) > 5:
            console.print(
                f"[dim]... 외 {len(search_results) - 5}개의 결과가 더 있습니다.[/dim]\n"
            )

        return search_results

    except Exception as error:
        # 에러 발생 시 에러 메시지 출력
        console.print(f"[red]❌ 에러 발생:[/red] {str(error)}")
        return None


def visualize_extract_contexts_and_images_function(
    evaluator: ContextRecallEvaluator, search_results: List[Dict]
):
    """
    _extract_contexts_and_images 함수의 동작을 시각화합니다.

    검색 결과에서 텍스트와 이미지 ID를 추출하는 과정을 보여줍니다.
    """
    # 섹션 제목 출력
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print(
        "📤 [bold cyan]함수 테스트: _extract_contexts_and_images()[/bold cyan]",
        style="bold",
    )
    console.print("=" * 80 + "\n", style="bold cyan")

    # 입력 데이터 통계 출력
    console.print(f"[yellow]입력 검색 결과 개수:[/yellow] {len(search_results)}개\n")

    try:
        # 추출 실행
        contexts, image_ids = evaluator._extract_contexts_and_images(search_results)

        # 추출 결과 통계
        console.print("[green]✓ 추출 완료[/green]")
        console.print(f"  - 컨텍스트 개수: {len(contexts)}개")
        console.print(f"  - 중복 제거된 이미지 ID 개수: {len(image_ids)}개\n")

        # 컨텍스트 목록을 패널로 표시
        contexts_panel_content = ""
        for index, context in enumerate(contexts[:3]):  # 처음 3개만
            preview = context[:80] + "..." if len(context) > 80 else context
            contexts_panel_content += f"[cyan]{index + 1}.[/cyan] {preview}\n"

        if len(contexts) > 3:
            contexts_panel_content += (
                f"\n[dim]... 외 {len(contexts) - 3}개의 컨텍스트가 더 있습니다.[/dim]"
            )

        console.print(
            Panel(
                contexts_panel_content, title="📝 추출된 컨텍스트", border_style="green"
            )
        )

        # 이미지 ID 목록을 패널로 표시
        image_ids_content = ", ".join(image_ids[:10])
        if len(image_ids) > 10:
            image_ids_content += f" ... (외 {len(image_ids) - 10}개)"

        console.print(
            Panel(image_ids_content, title="🖼️ 추출된 이미지 ID", border_style="magenta")
        )
        console.print()

        return contexts, image_ids

    except Exception as error:
        # 에러 발생 시 에러 메시지 출력
        console.print(f"[red]❌ 에러 발생:[/red] {str(error)}")
        return None, None


def visualize_compute_average_recall_function(
    evaluator: ContextRecallEvaluator, ragas_dataset: List[Dict]
):
    """
    _compute_average_recall 함수의 동작을 시각화합니다.

    RAGAS를 사용하여 컨텍스트 리콜을 계산하는 과정을 보여줍니다.
    """
    # 섹션 제목 출력
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print(
        "📊 [bold cyan]함수 테스트: _compute_average_recall()[/bold cyan]", style="bold"
    )
    console.print("=" * 80 + "\n", style="bold cyan")

    # 입력 데이터셋 정보 출력
    console.print(
        f"[yellow]평가할 데이터셋 크기:[/yellow] {len(ragas_dataset)}개 샘플\n"
    )

    # 데이터셋 샘플 미리보기
    if ragas_dataset:
        sample = ragas_dataset[0]
        console.print("[dim]데이터셋 샘플 구조:[/dim]")
        console.print(f"  - user_input: {sample.get('user_input', 'N/A')[:60]}...")
        console.print(
            f"  - retrieved_contexts: {len(sample.get('retrieved_contexts', []))}개"
        )
        console.print(f"  - reference: {sample.get('reference', 'N/A')[:60]}...\n")

    try:
        # RAGAS 평가 실행 (시간이 걸릴 수 있음)
        console.print(
            "[yellow]⏳ RAGAS 평가 실행 중...[/yellow] (이 작업은 시간이 걸릴 수 있습니다)"
        )

        average_recall = evaluator._compute_average_recall(ragas_dataset)

        # 결과를 큰 패널로 표시
        result_content = f"""
[bold green]평균 Context Recall:[/bold green] {average_recall:.4f}

[dim]Context Recall 지표 설명:[/dim]
- 값 범위: 0.0 ~ 1.0
- 높을수록 좋음 (1.0이 최고)
- 의미: 정답(ground truth)에 있는 정보가 검색된 컨텍스트에 
  얼마나 잘 포함되어 있는지를 측정
        """

        console.print(
            Panel(
                result_content.strip(), title="✅ 평가 결과", border_style="bold green"
            )
        )
        console.print()

        return average_recall

    except Exception as error:
        # 에러 발생 시 에러 메시지 출력
        console.print(f"[red]❌ 에러 발생:[/red] {str(error)}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None


def visualize_full_evaluation_pipeline(jsonl_path: str, top_k: int = 5):
    """
    전체 평가 파이프라인을 시각화합니다.

    evaluate_context_recall_at_k 함수의 전체 실행 흐름을 단계별로 보여줍니다.
    """
    # 메인 제목 출력
    console.print("\n" + "=" * 80, style="bold yellow")
    console.print(
        "🚀 [bold yellow]전체 파이프라인 테스트: evaluate_context_recall_at_k()[/bold yellow]",
        style="bold",
    )
    console.print("=" * 80 + "\n", style="bold yellow")

    try:
        # Evaluator 초기화
        console.print("[yellow]⏳ ContextRecallEvaluator 초기화 중...[/yellow]")
        evaluator = ContextRecallEvaluator()
        console.print("[green]✓ 초기화 완료[/green]\n")

        # 평가 실행
        console.print(f"[yellow]⏳ Top-{top_k} 평가 실행 중...[/yellow]")
        average_recall, details, ragas_dataset, all_image_ids = (
            evaluator.evaluate_context_recall_at_k(jsonl_path, top_k=top_k)
        )

        # 최종 결과를 테이블로 표시
        result_table = Table(
            title="📊 최종 평가 결과", show_header=True, header_style="bold cyan"
        )
        result_table.add_column("항목", style="yellow", width=30)
        result_table.add_column("값", style="green", width=40)

        # 테이블에 결과 추가
        result_table.add_row("평균 Context Recall@K", f"{average_recall:.4f}")
        result_table.add_row("K 값", str(details["k"]))
        result_table.add_row("평가 샘플 수", str(details["num_samples"]))
        result_table.add_row("생성된 RAGAS 데이터셋 크기", f"{len(ragas_dataset)}개")
        result_table.add_row("이미지 ID 리스트 개수", f"{len(all_image_ids)}개")

        console.print(result_table)
        console.print()

        # 성능 해석 출력
        console.print(
            Panel(
                _interpret_recall_score(average_recall),
                title="💡 결과 해석",
                border_style="blue",
            )
        )
        console.print()

        return average_recall, details, ragas_dataset, all_image_ids

    except Exception as error:
        # 에러 발생 시 에러 메시지 출력
        console.print(f"[red]❌ 에러 발생:[/red] {str(error)}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None


def _interpret_recall_score(score: float) -> str:
    """
    리콜 점수를 해석하여 설명 문자열을 반환합니다.

    점수 범위에 따라 성능 수준을 평가합니다.
    """
    if score >= 0.9:
        return "[bold green]🎉 우수[/bold green]\n검색된 컨텍스트가 정답을 매우 잘 포함하고 있습니다."
    elif score >= 0.7:
        return "[bold blue]👍 양호[/bold blue]\n검색된 컨텍스트가 정답을 잘 포함하고 있습니다."
    elif score >= 0.5:
        return "[bold yellow]⚠️ 보통[/bold yellow]\n검색된 컨텍스트에 정답이 부분적으로만 포함되어 있습니다."
    else:
        return "[bold red]❌ 개선 필요[/bold red]\n검색된 컨텍스트에 정답이 충분히 포함되어 있지 않습니다."


def run_all_tests(jsonl_path: str):
    """
    모든 시각화 테스트를 순차적으로 실행합니다.

    각 함수의 동작을 단계별로 시각화하여 보여줍니다.
    """
    # 프로그램 시작 메시지
    console.print("\n" + "🔬" * 40, style="bold magenta")
    console.print(
        "[bold magenta]Context Recall Evaluator 함수 시각화 테스트 프로그램[/bold magenta]",
        justify="center",
    )
    console.print("🔬" * 40 + "\n", style="bold magenta")

    # 1. load_test_data 함수 테스트
    samples = visualize_load_test_data_function(jsonl_path)

    if not samples:
        console.print(
            "[red]⚠️ 테스트 데이터를 로드할 수 없어 다른 테스트를 건너뜁니다.[/red]"
        )
        return

    # 간단한 목업 데이터로 개별 함수 테스트
    console.print("\n[yellow]━━━ 개별 함수 테스트 시작 ━━━[/yellow]\n")

    try:
        # Evaluator 초기화
        console.print("[yellow]⏳ ContextRecallEvaluator 초기화 중...[/yellow]")
        evaluator = ContextRecallEvaluator()
        console.print("[green]✓ 초기화 완료[/green]\n")

        # 2. _build_evaluator_llm 함수 테스트
        visualize_build_evaluator_llm_function(evaluator)

        # 3. _search_contexts 함수 테스트
        test_query = samples[0]["question"] if samples else "What is machine learning?"
        search_results = visualize_search_contexts_function(
            evaluator, test_query, top_k=5
        )

        if search_results:
            # 4. _extract_contexts_and_images 함수 테스트
            contexts, image_ids = visualize_extract_contexts_and_images_function(
                evaluator, search_results
            )

            # 5. _compute_average_recall 함수 테스트 (간단한 목업 데이터 사용)
            if contexts:
                mock_ragas_dataset = [
                    {
                        "user_input": test_query,
                        "retrieved_contexts": contexts,
                        "reference": samples[0].get("ground_truth", "Test reference"),
                    }
                ]

                # 주의: 이 함수는 실제 API 호출을 하므로 비용이 발생할 수 있습니다
                console.print(
                    "[yellow]⚠️ _compute_average_recall 테스트는 실제 API를 호출합니다.[/yellow]"
                )
                user_input = typer.prompt("계속 진행하시겠습니까? (y/n)")

                if user_input.lower() == "y":
                    visualize_compute_average_recall_function(
                        evaluator, mock_ragas_dataset
                    )
                else:
                    console.print(
                        "[dim]_compute_average_recall 테스트를 건너뜁니다.[/dim]\n"
                    )

        # 6. 전체 파이프라인 테스트
        console.print("\n[yellow]━━━ 전체 파이프라인 테스트 시작 ━━━[/yellow]\n")
        console.print(
            "[yellow]⚠️ 이 테스트는 실제 API를 호출하여 비용이 발생할 수 있습니다.[/yellow]"
        )
        user_input = typer.prompt("전체 파이프라인 테스트를 실행하시겠습니까? (y/n)")

        if user_input.lower() == "y":
            visualize_full_evaluation_pipeline(jsonl_path, top_k=5)
        else:
            console.print("[dim]전체 파이프라인 테스트를 건너뜁니다.[/dim]\n")

    except Exception as error:
        # 최상위 에러 핸들링
        console.print(f"[red]❌ 치명적 에러 발생:[/red] {str(error)}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")

    # 프로그램 종료 메시지
    console.print("\n" + "🎬" * 40, style="bold magenta")
    console.print("[bold magenta]테스트 프로그램 종료[/bold magenta]", justify="center")
    console.print("🎬" * 40 + "\n", style="bold magenta")


# Typer 커맨드 정의


@app.command("all")
def cmd_all(
    jsonl: str = typer.Option(
        "/home/kun/Desktop/multimodal/data/test_samples.jsonl",
        "--jsonl",
        help="테스트 데이터 JSONL 파일 경로",
    ),
):
    """
    모든 함수 테스트를 순차적으로 실행합니다.

    전체 시각화 테스트를 한 번에 실행하여 각 함수의 동작을 확인합니다.
    """
    run_all_tests(jsonl)


@app.command("load-data")
def cmd_load_data(
    jsonl: str = typer.Option(
        ...,
        "--jsonl",
        help="로드할 JSONL 파일 경로",
    ),
):
    """
    load_test_data() 함수를 테스트합니다.

    JSONL 파일에서 데이터를 읽어오는 과정을 시각화합니다.
    """
    console.print("\n[bold cyan]📂 데이터 로드 테스트 시작[/bold cyan]\n")
    visualize_load_test_data_function(jsonl)


@app.command("build-llm")
def cmd_build_llm():
    """
    _build_evaluator_llm() 함수를 테스트합니다.

    OpenAI LLM 객체를 생성하는 과정을 확인합니다.
    """
    console.print("\n[bold cyan]🤖 LLM 빌더 테스트 시작[/bold cyan]\n")
    console.print("[yellow]⏳ ContextRecallEvaluator 초기화 중...[/yellow]")
    evaluator = ContextRecallEvaluator()
    console.print("[green]✓ 초기화 완료[/green]\n")
    visualize_build_evaluator_llm_function(evaluator)


@app.command("search")
def cmd_search(
    query: str = typer.Option(
        ...,
        "--query",
        help="검색할 질문 또는 쿼리",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        help="검색할 결과 개수",
    ),
):
    """
    _search_contexts() 함수를 테스트합니다.

    벡터 스토어에서 쿼리와 유사한 문서를 검색하는 과정을 시각화합니다.
    """
    console.print("\n[bold cyan]🔍 검색 테스트 시작[/bold cyan]\n")
    console.print("[yellow]⏳ ContextRecallEvaluator 초기화 중...[/yellow]")
    evaluator = ContextRecallEvaluator()
    console.print("[green]✓ 초기화 완료[/green]\n")
    visualize_search_contexts_function(evaluator, query, top_k)


@app.command("extract")
def cmd_extract(
    query: str = typer.Option(
        ...,
        "--query",
        help="검색 후 추출할 쿼리",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        help="검색할 결과 개수",
    ),
):
    """
    _extract_contexts_and_images() 함수를 테스트합니다.

    검색 결과에서 텍스트와 이미지 ID를 추출하는 과정을 시각화합니다.
    """
    console.print("\n[bold cyan]📤 데이터 추출 테스트 시작[/bold cyan]\n")
    console.print("[yellow]⏳ ContextRecallEvaluator 초기화 중...[/yellow]")
    evaluator = ContextRecallEvaluator()
    console.print("[green]✓ 초기화 완료[/green]\n")

    # 먼저 검색 수행
    search_results = evaluator._search_contexts(query, top_k)
    console.print(f"[green]✓ 검색 완료:[/green] {len(search_results)}개의 결과\n")

    # 추출 시각화
    visualize_extract_contexts_and_images_function(evaluator, search_results)


@app.command("recall")
def cmd_recall(
    jsonl: str = typer.Option(
        ...,
        "--jsonl",
        help="평가할 데이터셋 JSONL 파일 경로",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        help="검색할 컨텍스트 개수",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="API 호출 확인 메시지를 건너뜁니다",
    ),
):
    """
    _compute_average_recall() 함수를 테스트합니다. (⚠️ API 비용 발생)

    RAGAS를 사용하여 컨텍스트 리콜을 계산하는 과정을 시각화합니다.
    실제 OpenAI API를 호출하므로 비용이 발생할 수 있습니다.
    """
    console.print("\n[bold cyan]📊 리콜 계산 테스트 시작[/bold cyan]\n")

    # API 호출 확인
    if not no_confirm:
        console.print(
            "[yellow]⚠️ 이 테스트는 실제 OpenAI API를 호출하여 비용이 발생합니다.[/yellow]"
        )
        user_input = typer.prompt("계속 진행하시겠습니까? (y/n)")
        if user_input.lower() != "y":
            console.print("[dim]테스트를 취소했습니다.[/dim]")
            raise typer.Exit()

    # 데이터 로드
    samples = load_test_data(jsonl)
    if not samples:
        console.print("[red]❌ 데이터를 로드할 수 없습니다.[/red]")
        raise typer.Exit(code=1)

    # Evaluator 초기화
    console.print("[yellow]⏳ ContextRecallEvaluator 초기화 중...[/yellow]")
    evaluator = ContextRecallEvaluator()
    console.print("[green]✓ 초기화 완료[/green]\n")

    # 첫 번째 샘플로 테스트
    test_query = samples[0]["question"]
    search_results = evaluator._search_contexts(test_query, top_k)
    contexts, _ = evaluator._extract_contexts_and_images(search_results)

    # RAGAS 데이터셋 구성
    mock_ragas_dataset = [
        {
            "user_input": test_query,
            "retrieved_contexts": contexts,
            "reference": samples[0].get("ground_truth", "Test reference"),
        }
    ]

    # 리콜 계산
    visualize_compute_average_recall_function(evaluator, mock_ragas_dataset)


@app.command("pipeline")
def cmd_pipeline(
    jsonl: str = typer.Option(
        ...,
        "--jsonl",
        help="평가할 데이터셋 JSONL 파일 경로",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="검색할 컨텍스트 개수",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="API 호출 확인 메시지를 건너뜁니다",
    ),
):
    """
    evaluate_context_recall_at_k() 전체 파이프라인을 테스트합니다. (⚠️ API 비용 발생)

    전체 평가 파이프라인을 실행하여 Context Recall@K를 계산합니다.
    실제 OpenAI API를 호출하므로 비용이 발생할 수 있습니다.
    """
    console.print("\n[bold cyan]🚀 전체 파이프라인 테스트 시작[/bold cyan]\n")

    # API 호출 확인
    if not no_confirm:
        console.print(
            "[yellow]⚠️ 이 테스트는 실제 OpenAI API를 호출하여 비용이 발생합니다.[/yellow]"
        )
        user_input = typer.prompt("계속 진행하시겠습니까? (y/n)")
        if user_input.lower() != "y":
            console.print("[dim]테스트를 취소했습니다.[/dim]")
            raise typer.Exit()

    visualize_full_evaluation_pipeline(jsonl, top_k)


if __name__ == "__main__":
    app()

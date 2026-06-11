# Assignment 4 Rules

기준 문서: `assignment4.pdf`

## 핵심

- 주제: `Neural Network Verification with α,β-CROWN`
- 도구: `α,β-CROWN`
- 대상 모델:
  - `alpha-beta-CROWN/complete_verifier/models`에 이미 포함되지 않은 외부 모델
  - `ONNX` 또는 `PyTorch` 형식 모델
- 핵심 개념:
  - `Bound Propagation`
  - `Linear Relaxation`
  - `Branch-and-Bound (BaB)`
  - `ℓ∞-ball` 입력 제약 하의 adversarial robustness verification
  - 검증 결과 `verified` / `falsified` / `timeout` 해석
- 실행 파일: `test.py`
- 최종 제출물:
  - `requirements.txt` 또는 `Dockerfile` 또는 `environment.yml`
  - `test.py`
  - YAML configuration file(s)
  - `report.pdf`
  - `README.md`

## 일정 / 제출

- 배포일: `Tuesday, June 2, 2026`
- 마감일: `Friday, June 12, 2026, 11:59 PM`
- 제출 방식: GitHub 업로드 후 Uclass에 repository link 제출
- 지각 제출: 시스템 / 이메일 모두 불가

## 프로젝트 구성

- 재현 가능한 환경 설정 포함
  - `requirements.txt`, `Dockerfile`, 또는 `environment.yml`
  - 외부 모듈 / dependency / Conda 환경 설정 기록
- 코드 이해용 주석 포함
- `test.py` 포함
  - 선택한 외부 모델에 대해 `α,β-CROWN` 실행
  - 검증 결과 출력
  - 실행 형식은 자유지만 코드의 타당성을 확인할 수 있어야 함
- YAML configuration file 포함
  - 모델, 데이터셋, 검증 속성, solver parameter 명시
- 재현성 확보
  - 외부 모델 파일을 포함하거나 모델 생성 / 변환 스크립트 제공
  - 검증에 필요한 데이터 샘플을 포함하거나 데이터 준비 스크립트 제공
  - `README.md`만 보고 설치와 실행 흐름을 이해할 수 있어야 함

## Problem 1: α,β-CROWN Models Directory 탐색

### 배경

- `α,β-CROWN`은 bound propagation, linear relaxation, branch-and-bound를 사용하는 신경망 검증 도구
- Assignment #3의 `Marabou`가 SMT 기반 접근을 사용하는 반면, `α,β-CROWN`은 bound 계산과 BaB를 통해 큰 네트워크에서 빠른 검증을 목표로 함
- `complete_verifier/models`와 `complete_verifier/exp_configs`를 살펴보며 제공 모델, YAML 설정 방식, 검증 구조를 이해해야 함

### 확인할 repository / directory

- GitHub repository:
  - `https://github.com/Verified-Intelligence/alpha-beta-CROWN`
- models directory:
  - `https://github.com/Verified-Intelligence/alpha-beta-CROWN/tree/main/complete_verifier/models`
- example config directory:
  - `complete_verifier/exp_configs/`

### 수행 내용

#### 1. 제공 모델 조사

- `complete_verifier/models`에 어떤 모델이 있는지 정리
- 각 모델에 대해 가능한 범위에서 다음 항목 기록:
  - 모델명 / 파일명
  - architecture
  - 파일 포맷
  - 입력 크기
  - 대상 데이터셋 또는 verification setup

#### 2. 제공 YAML configuration 조사

- `complete_verifier/exp_configs`의 example config 확인
- 각 config에 대해 가능한 범위에서 다음 항목 기록:
  - YAML 파일명
  - 대상 모델 / 데이터셋
  - 검증 속성
  - 주요 solver 설정
  - timeout, branching strategy, complete verification 사용 여부

#### 3. Marabou와 모델 명세 방식 비교

- 비교할 항목:
  - 기본 검증 접근
  - 지원 / 사용 모델 포맷
  - 검증 명세 방식
  - 설정 방식
  - 결과 해석 방식

- 핵심 비교 방향:
  - `Marabou`: SMT 기반, Python API나 query 구성 중심
  - `α,β-CROWN`: YAML configuration 중심, ONNX / PyTorch 모델 중심, bound propagation + BaB 기반
  - `Marabou` 결과: `SAT` / `UNSAT` / `TIMEOUT`
  - `α,β-CROWN` 결과: `verified` / `falsified` / `timeout`

## Problem 2: 외부 모델과 데이터셋으로 α,β-CROWN 실행

### 배경

- `α,β-CROWN`의 실제 사용 능력을 보이기 위해 repository의 built-in models directory에 없는 외부 모델을 선택해야 함
- 모델 준비, YAML 설정 작성, 검증 실행, 결과 해석까지 직접 수행해야 함
- Assignment #3에서 사용한 모델이 있다면 같은 모델과 property를 사용해 Marabou와 비교하는 것이 권장됨

### 수행 내용

#### 1. α,β-CROWN 설치

- repository clone

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
```

- Conda 환경 생성 권장
- 설치 후 provided example configuration 실행으로 동작 확인
- 설치 중 발생한 문제와 해결 방법 기록
  - CUDA / PyTorch version mismatch
  - missing package
  - path error
  - model / config 호환성 문제

#### 2. 외부 모델 및 데이터셋 선택

- `complete_verifier/models`에 이미 포함되지 않은 모델 선택
- 모델 형식은 `ONNX` 또는 `PyTorch`
- 작은 모델로 시작 권장
  - 너무 큰 CNN / ResNet은 complete verification에서 timeout 가능성이 큼
- 선택 이유를 report에 기록
- Assignment #3 모델을 재사용하는 경우:
  - 동일 모델 / 동일 데이터셋 / 동일 property 기준으로 Marabou와 비교 가능

#### 3. YAML configuration 작성

- `α,β-CROWN`은 YAML file을 통해 모델, 데이터셋, 입력 제약, 검증 속성, solver 설정을 지정함
- 실제 YAML key는 `complete_verifier/exp_configs`의 example config를 기준으로 맞출 것
- YAML에 포함해야 할 정보:
  - model path
  - model architecture 또는 format
  - dataset 또는 input sample path
  - input specification
  - perturbation radius `epsilon`
  - verification property
  - solver parameters
  - timeout
  - branching strategy

#### 4. 검증 속성 작성

- 가장 기본적인 검증 속성은 adversarial robustness
- 예시 property:
  - 입력 `x`의 label이 `y`일 때,
  - 모든 `x'`가 `||x' - x||∞ <= ε`을 만족하면,
  - 모델의 예측 label이 계속 `y`인지 검증
- 즉, `ℓ∞` ball 내부에서 label이 변하지 않는지 확인
- `epsilon` 값은 데이터 스케일과 정규화 여부를 고려해 설정
- `epsilon`이 너무 작으면 대부분 `verified`가 될 수 있음
- `epsilon`이 너무 크면 `falsified` 또는 `timeout`이 많아질 수 있음

#### 5. 실행 및 결과 해석

- 실행 방식 예시:

```bash
cd alpha-beta-CROWN/complete_verifier
python abcrown.py --config path/to/config.yaml
```

- 또는 wrapper를 제공하는 경우:

```bash
python test.py
```

- `test.py` 최소 요구:
  - 사용할 YAML config 경로 확인
  - `α,β-CROWN` 실행
  - 검증 결과 출력
  - runtime 출력
  - 가능하면 결과를 `results/` 폴더에 저장

- instance별로 기록할 항목:
  - sample index
  - true label 또는 target label
  - verification status
  - runtime
  - counterexample 존재 여부

### 결과 해석

- `verified`:
  - 주어진 입력 영역 전체에서 property가 성립
  - 해당 `epsilon` 범위 내 robustness 보장
- `falsified`:
  - property를 깨는 counterexample 발견
  - adversarial input 존재
- `timeout`:
  - 제한 시간 내 결론 미도달
  - `verified`도 `falsified`도 아님
- `error`:
  - 설정, 모델 경로, input shape, dependency 등의 문제 가능성 확인

## 채점 / 제출 결과

### 4.1 Exploration Report (`20%`)

- `complete_verifier/models` directory의 제공 모델 요약
- 모델 architecture와 format 정리
- 사용 가능한 YAML configuration 요약
- 제공 example들이 다루는 dataset / property 정리
- 가능하면 Marabou resource organization과 비교

### 4.2 Implementation & Results (`50%`)

- 제출해야 할 구현 내용:
  - `α,β-CROWN` 실행 코드
  - 외부 모델 파일 또는 생성 / 변환 스크립트
  - 데이터셋 샘플 또는 다운로드 / 준비 코드
  - YAML configuration file
  - verification result 출력
  - runtime 측정
  - counterexample이 있을 경우 저장 또는 설명

- 보고해야 할 내용:
  - 선택한 모델과 데이터셋 및 선택 이유
  - compatibility를 위해 필요한 수정 사항 또는 preprocessing
  - instance별 `verified` / `falsified` / `timeout` 결과
  - verification runtime
  - counterexample 발견 여부
  - Assignment #3과 같은 모델을 사용했다면 Marabou와 비교

### 4.3 Report (`30%`)

- `report.pdf`는 `1-2 pages`
- 포함할 내용:
  - 모델, 데이터셋, verification property 설명
  - 결과와 결과 해석
  - 같은 모델을 사용한 경우 `α,β-CROWN`과 `Marabou` 비교
    - speed
    - scalability
    - ease of use
    - supported features
  - 직접 사용해 본 관점에서 `α,β-CROWN`의 장점과 한계 논의

## AI 사용 / Git 히스토리

- AI assistant 사용 가능
- 단, 본인의 genuine understanding을 보여야 함
- GitHub repository는 meaningful commit history 필요
- single bulk commit 지양
- 채점자가 `git log`를 inspect할 수 있음
- report는 generic description이 아니라 직접 관찰한 실험 결과와 해석을 포함해야 함
- unexpected result가 있었다면 구체적으로 논의

## 최종 제출물

- `requirements.txt` 또는 `Dockerfile` 또는 `environment.yml`
- `test.py`
- YAML configuration file(s)
- `report.pdf`
- `README.md`

## README 요구사항

- `α,β-CROWN` 설치 방법
- 환경 설정 방법
- 외부 모델 / 데이터셋 준비 방법
- YAML config 위치와 의미
- `test.py` 실행 방법
- 예상 출력 또는 결과 파일 위치
- 재현을 위해 필요한 주의사항

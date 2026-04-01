# Detection-Attack: MMDetection 기반 객체 탐지 적대적 공격 파이프라인

이 저장소는 MMDetection 프레임워크를 기반으로 동작하는 **적대적 공격(Adversarial Attack)** 파이프라인을 포함하고 있습니다.
주로 모델의 픽셀 단위 조작 취약성을 찾는 $L_0$ 최적화 기반의 Black-Box/Decision-based 공격 알고리즘(SparseEvo, PointWise)을 지원합니다.

---

## 🚀 주요 기능 (Features)
- **여러 가지 공격 알고리즘 제공:**
  - `sparse_evo`: 진화 연산(Evolutionary Algorithm)을 활용하여 적은 쿼리로 스파스(Sparse)한 노이즈를 텍스처링 탐색합니다.
  - `pointwise`: 1픽셀 단위로 교체해가며 이진 탐색(Binary Search)을 통해 정밀한 픽셀 경계값을 찾아냅니다.
  - `pointwise_multi`: `npix`(픽셀 그룹) 단위로 묶어서 탐색 효율을 높인 배포형 PointWise 입니다.
  - `pointwise_multi_sched`: 큰 픽셀 그룹부터 시작하여 점진적으로 크기를 절반씩 줄여나가는 스케줄링이 도입된 고성능 알고리즘입니다.
- **다중 프레임워크 지원:**
  - **MMDetection**: `rtmdet`, `ddq-detr` 등 기존 MMDetection 기반 모델 지원 (`--model-type mmdet`)
  - **Ultralytics YOLOv8**: YOLOv8 모델 직접 연동 지원. 별도의 체크포인트가 없을 경우 Ultralytics가 가중치를 자동 다운로드하여 구성합니다 (`--model-type yolov8`).
- **다양한 형태의 Ground Truth 평가 지원:**
  - **COCO 포맷 (`.json`)**: 단일 어노테이션 파일 기반 검증
  - **YOLO / Argoverse 포맷 (`.txt`)**: 각 이미지 파일명과 매칭되는 `class_id xc yc w h` (Normalized Ratio [0, 1]) 디렉터리를 직접 인식하여 내부 모델 사이즈로 자동 정규화합니다.
- **풍부한 결과물(Visualization & Stats):**
  - 원본 Bbox, 공격된 Bbox (Confidence 포함) 이미지 출력 (`orig.png`, `adv.png`)
  - 공격이 수행된 순수 노이즈가 더해진 맵 (`delta.png`, `adv_raw.png`)
  - 실시간 실험 과정 및 쿼리 예산을 확인할 수 있는 `experiment_report.txt` 및 CSV 출력

---

## 🛠️ 환경 구성 및 실행 방법

### 1. Docker 환경 세팅 (필수)
이 파이프라인은 복잡한 의존성을 피하기 위해 Docker Compose를 활용한 환경 구성을 권장합니다.
저장소 루트에 있는 `docker-compose.yaml`을 통해 컨테이너를 올리고 접속해 주세요.

```bash
# 컨테이너 빌드 및 백그라운드 실행
docker-compose up -d --build
```

> ⚠️ **주의:** 컨테이너가 최초로 실행될 때 내부적으로 패키지 의존성 설치(`pip install`) 등 기초 세팅을 진행하여 시간이 다소 소요됩니다. 
> 곧바로 접속하지 마시고, 아래 명령어를 통해 내부 설치 로그가 끝나는 것을 확인하고 접속하세요!

```bash
# 설치 등의 백그라운드 로그 실시간 확인 (Ctrl+C로 종료)
docker logs -f mmdet

# 설치 완료 확인 후 컨테이너 쉘 접근
docker-compose exec mmdet bash
```
*(성공적으로 터미널에 접속하신 이후, 아래의 스크립트 실행 과정을 진행하시면 됩니다.)*

### 2. 스크립트 설정 및 실행
파이프라인을 가장 쉽게 실행하려면 미리 템플릿으로 작성된 실행 방식(`*.bash`) 파일을 사용하시면 됩니다. 현재 2가지 버전을 제공합니다.

#### A. MMDetection 모델(기본) 기반 공격 (`run.bash`)
가장 일반적인 형태입니다. 내부에서 사용할 모델과 이미지, 라벨을 세팅합니다.
```bash
# 파일명: run.bash

CONFIG=configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py
CHECKPOINT=ckpt/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth
IMAGE_DIR="data/coco_amnesia/val2017" # 디렉토리 내를 재귀적으로 검색합니다.
ANN_FILE="data/coco_amnesia/instances_val2017_ori.json"
```

터미널에서 아래명령어를 쳐서 실행합니다.
```bash
bash run.bash
```

#### B. YOLOv8 최신 모델 기반 공격 (`run_yolo.bash`)
Ultralytics YOLO 기반으로 동작하며, 모델 체크포인트를 설정하여 공격을 수행할 수 있습니다.
미리 YOLO 가중치를 `ckpt` 폴더에 다운로드 받아 사용하는 것을 권장합니다.

```bash
# 가중치 파일 다운로드 (ckpt 폴더 하위로)
mkdir -p ckpt
curl -L -o ckpt/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
```

```bash
# 파일명: run_yolo.bash

MODEL_TYPE="yolov8"
CHECKPOINT="ckpt/yolov8n.pt"  # ckpt 하위에 저장된 모델 가중치 경로
IMAGE_DIR="data/Argoverse_sample1_amnesia" # 재귀적으로 하위 `images/val`을 자동 추적
ANN_FILE="data/Argoverse_sample1_amnesia/labels/val" # YOLO 포맷 폴더를 주면 자동으로 파싱
```

터미널에서 아래명령어를 쳐서 실행합니다.
```bash
bash run_yolo.bash
```

---

## 📊 결과물 및 확인
공격이 종료되면 다음과 같은 체계적인 파일들이 `result/[attack_method]/[model_name]/[시간]/` 하위에 생성됩니다.

1. `experiment_report.txt`: 현재 실험의 조건, 평균 소진 쿼리(Query), L0 Sparsity 수치, 공격 성공률, 벤치마크 모델과 GT 대비 mAP 하락률이 일목요연하게 표시됩니다.
2. `attack_results_*.csv`: 이미지 하나하나의 개별적인 성적표가 저장됩니다.
3. `images/[이미지이름]/` 폴더 내:
   - `query_0.png` ~ `query_[N].png`: `max_query / 5` 간격으로 촬영된 진화/탐색하는 노이즈 상태 스냅샷 저장
   - `orig.png`: 모델이 바라본 원본 이미지와 예측된 Bounding Box (기준점)
   - `adv.png`: 공격당한 후의 이미지와 속아 넘어간 Bounding Box 결과 (Misclassified, Disappeared, Survived 등 추적용)
   - `adv_raw.png`: 박스가 그려지지 않은 순수 적대적 이미지
   - `delta.png`: 원본과 공격 이미지간 바뀐 픽셀을 가장 강조한 노이즈 차이 맵 (Heatmap)

---

## 🔬 공격 파라미터 튜닝 방법
`run.bash` 내의 하단 인자들을 조절하여 탐색 효율을 바꿀 수 있습니다.
- **SparseEvo 계열:**
  - `--pop-size`: 한 번의 변이 세대에 사용할 개체군 크기 (기본값 10)
  - `--cr`, `--mu`: 변이 탐색률 및 난수 반영 확률
- **PointWise 계열:**
  - `--npix`: 한번에 교체를 시도할 픽셀 비율 또는 개수 (기본값 `0.1` 이면 전체 해상도의 10% 픽셀씩 블록 교체 시도)

> 💡 **Tip:** PointWise 공격의 두 번째 이진 탐색(Phase 2)은 언제나 남은 쿼리 예산(`max_query`)을 초과하지 않는 선에서 최적화를 종료하도록 완전하게 제어되므로 안심하고 Budget 한도를 타이트하게 정하셔도 됩니다.

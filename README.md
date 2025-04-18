# 🦜 패럿 데이터사이언스 학회 — 팀 3조 이미지 분류 프로젝트

패럿 데이터사이언스 학회 3조가 수행한 **이미지 분류 모델 개발** 프로젝트입니다.  
다양한 데이터 전처리와 고급 Augmentation 기법을 적용해 모델의 일반화 성능을 극대화했고, **EfficientNet‑B1** 기반 전이학습을 활용하여 최적의 결과를 얻었습니다.

---

## 📂 저장소 구조

. ├── 3조.ipynb # 전체 분석·실험 기록이 담긴 Jupyter Notebook ├── label_table_3조.csv # 클래스 라벨 매핑 정보 CSV ├── train.py # 모델 학습 스크립트 (PyTorch) ├── requirements.txt # 필요한 라이브러리 목록 └── README.md # 이 설명 문서


---

## 🎯 프로젝트 개요

- **목표**  
  - 이미지 데이터셋을 전처리·증강하여 모델 일반화 성능을 개선  
  - 전이학습(pretrained EfficientNet‑B1) + fine‑tuning  
  - 스케줄러, Early Stopping을 통한 안정적 학습

- **주요 사용 기술**  
  - Python, PyTorch, torchvision, timm  
  - Jupyter Notebook  
  - 데이터 증강: RandomResizedCrop, ColorJitter, Gaussian Noise 등

---

## 🛠 주요 구성 요소

### 1. 데이터 전처리 & Augmentation

```python
# 사용자 정의 Gaussian Noise 추가 클래스
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean, self.std = mean, std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    Train Transform

        RandomResizedCrop(224)

        RandomHorizontalFlip(), RandomVerticalFlip()

        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

        RandomRotation(±20°), RandomAffine(translate=0.1)

        ToTensor() → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

        AddGaussianNoise(mean=0.0, std=0.1) (커스텀 노이즈 증강)

    Validation Transform

        Resize(256) → CenterCrop(224)

        ToTensor() → Normalize(...)

    노고 포인트:
    다양한 공간·색상 변환과 Gaussian Noise를 조합해 모델이 잡음과 왜곡에 강건하도록 학습했습니다.

2. 모델 아키텍처

    기반 모델: efficientnet_b1 (timm 라이브러리)

    Classifier 수정:

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )

3. 학습 파이프라인

    DataLoader

        batch_size=64, shuffle=True (Train), shuffle=False (Val)

        num_workers=2

    Optimizer & Scheduler

        optimizer = AdamW(lr=1e-4, weight_decay=0.01)

        scheduler = ReduceLROnPlateau(factor=0.1, patience=10)

    Loss & Metrics

        criterion = CrossEntropyLoss()

        Accuracy 계산

    Early Stopping

        검증 손실이 개선되지 않으면 patience=10 이후 학습 중단

    Training Loop

    for epoch in range(num_epochs):
        # train / val 모드 전환
        # 배치 단위 순전파·역전파 / 평가
        # 손실 & 정확도 로깅
        # scheduler.step(val_loss)
        # early stopping 체크

4. 결과 저장

    최종 모델: /home/work/team3/model2.pt

    torch.save(model.state_dict(), save_path) 로 학습된 가중치 저장

🚀 사용 방법

    저장소 클론

git clone https://github.com/your-org/team3-image-classification.git
cd team3-image-classification

의존성 설치

pip install -r requirements.txt

데이터셋 준비

/dataset
├── train/  # 클래스별 서브디렉터리
└── valid/

학습 실행

    python train.py \
      --data_dir /path/to/dataset \
      --epochs 100 \
      --batch_size 64 \
      --save_path ./model2.pt

✨ 향후 과제

    추가 Augmentation 효과 비교 실험

    다양한 Backbone(EfficientNet‑B, ResNet, ViT) 벤치마킹*

    앙상블 및 하이퍼파라미터 최적화

    모델 배포(API/웹 서비스화)


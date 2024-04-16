# Intelligent Data Analysis & Machine Learning

## Supervised Learning

### Classification
강의 1 초반부, 채워야함
### Regression

### Ordinal Regression

### Metric Learning

Metric Learning은 간략하게, 데이터 간의 유사도를 잘 수치화하는 거리 함수(metric function)를 학습하는 것이다. 여기에 쓰일 수 있는 metric으로는 유클리디언 다스탠스가 있겠지만, 몇가지 단점 때문에 고차원에서는 사용하기 어렵다. 유클리드 거리는 클래스 간의 상관관계를 고려하지 않고 isotropic(모든 방향에서 동일) 하다.이에 우리는 차원 간 관계를 캡처하는 non-isotropic 거리를 사용할 수 있다.

미리 정의한 Distance Metric들은 데이터와 task에 대한 고려가 적기 떄문에 우리 용도에 적합하지 않다.(Mahalanobis Distance조차도 공분산을 고려한 Linear Transformation을 한 것일 뿐이다)

이러한 이유로 데이터에 적합한 거리 함수를 기계학습으로 직접 만드는 것이 Metric learning이며, 기계학습 중에서도 딥러닝을 활용하는 경우가 Deep Metric Learning이다.

Metric Learning의 목표는 데이터로부터 Metric function을 학습하는 것이다.

이를 달성하고자 보통 우리는 original feature space를 거리 계산하기 쉬운 embedding space로 mapping하는 embedding 함수 f를 학습한다. 이때 학습된 metric function은 아래에서 보이듯이 similiarity를 나타내는 s일수도 있고, distance를 나타내는 d일수도 있다.

Metric Learning은 Classification 문제에서 새로운 클래스를 식별하는 것을 목표로 하는 방법 중 하나이다. 일반적으로 Classification problem에서는 training time에 이미 정의되고 고정된 classes가 있다.    
그러나 일부 model에서는 이전에 본 적 없는 클래스를 식별할 수 있어야 한다. 예를 들어, 생체 인식에서는 모든 개인의 얼굴 이미지, 지문 스캔, 홍채 스캔을 하나의 클래스로 볼 수 있다. 새로운 개인은 그에 따라 인식되어야 한다. 이게 무슨 말이냐면, 이제 분류를 하려고 머신러닝을 시키는데, 매일 분류 대상이 바뀐다는 것이다. 이걸 해결한느 것이 metric learning이다.

또 다른 예시로는 object classification이 있다. 여기서는 새로운, 이전에 본 적이 없는 유형의 object image가 동일한 클래스에 속한다고 인식되어야 한다.

Metric learning은 **similarity metric s(z,z')**에 기반하여 데이터 인스턴스 x $\in X$를 embedding vector z로 매핑하는 방법이다. 메트릭 러닝에서 learner에서 제공되는 input은 다음과 같다:

- Training instances x with classes y.
- $T_n = \{(x_1,y_1),\dots,(x_n,y_n)\}$

이는 위의 classification에서 이미 봤던 것과 비슷하다.

그러나 output의 경우는 조금 다르다. 모델은 input instances를 embedding vector에 assign해야 한다.  함수 $f: x\mapsto z$로 표시되는 모델로써 이 모델은 입력데이터 x를 embedding vector z로 변환하는 역할을 한다.

이에 따라 learner는 다음과 같은 output을 생성해야한다. 

- 동일한 클래스 y에 속하는 모든 인스턴스 x, x'에 대해 similarity가 유지된다:
  - $s(f_{\Theta}(\mathbf{x}),f_{\Theta}(\mathbf{x'})) > \varepsilon$
  - 즉, 같은 클래스 내의 두 vector는 서로 similar properties를 가져야 한다는 것이다. 여기서 $f_{\Theta}$는 instance의 embedding이다.
  - 예를 들면, 비슷한 두 사람을 담은 두개의 사진이 있을 때, 그 둘에 대한 두 embedding vectors의 similarity는 $\varepsilon$보다, 즉 특정 값보다 커야한다는 것이다.
- 서로다른 클래스 y, y'에 속하는 인스턴스 x, x'는 서로 다른 클래스에 속한다는 것을 보장하기 위해 dissimilarity가 유지된다:
  - $s(f_{\Theta}(\mathbf{\tilde{x}}),f_{\Theta}(\mathbf{\tilde{x}'})) < \varepsilon$
  - 위에 틸드가 들어간 x는 dissimilarity를 나타내기  위해 사용된다.

Classification과 비교해서, metric learning은 장점을 가진다.

1. 일반적인 Classification은 학습한 이미지에 대해서만, 인식이 가능하다.

     - 반면, Metric Learning은 학습하지 않은 이미지도 DB로 구축만 해둔다면 인식 가능하다.

2. 일반적인 Classification은 Feature 간의 Decision Boundary를 찾도록 학습한다.(Learn separable features)

    - 반면, Metric Learning은 비유사한 Feature들을 멀리 떨어지도록 학습한다.(Learn Large-Margin features) Feature 공간을 더 잘 사용할 수 있음
### Taxonomy Classification

마찬가지로 input은 instance x $\in X$로 주어진다.

일반적인 classification처럼 discrete value $y \in Y$가 output으로 생성되지만, 여기선 일반적인 classification과는 달리, Y의 elements에 대해 트리 기반의 ordering이 주어진다. 

이러한 방식은 예측된 값과 실제 값 사이의 거리가 멀 수록 prediction이 나빠진다는 특징을 가진다. 이는 카테고리 간의 순서가 있기 때문에 발생한다. 

Parse natural languages 혹은 Protein folding의 prediction에 사용될 수 있다.

### Rankings

Ranking에서 input은 query $q$와 아이템 리스트 $I_1,\dots,I_n$으로 주어진다. output은 sorted item list이다. 

사용되는 Training data는 $q$로 쿼링 한 이후 유저가 $I_j$를 클릭한 것이다. 클릭된 아이템은 클릭되지 않은 아이템보다 높은 순위로 나열되어야 한다.  

이는 보통 검색 엔진에서 사용된다. 사용자의 피드백을 통해 아이템들을 정렬하고, 사용자가 선택한 아이템을 클릭한 후 해당 아이템의 순위를 높이는 것이 목표이다.

### Recommendations

- Input: users, items, contextual information
- Output: 사용자가 recommandation을 얼마나 좋아할지에 대한 prediction
- Training data: ratings, sales, page views 등에 대한 정보

사용자에게 관련성이 높은 아이템을 추천하여 사용자 만족도를 향상시키고, 아이템의 판매량을 증가시키는 것이 목표이다. 트레이닝 데이터는 사용자가 아이템에 대해 평가한 등급이나 실제 구매 행동 등을 기반으로 한다. 이를 통해 사용자의 취향과 관심사를 파악하고, 이에 맞춰서 개인화된 추천을 제공한다.

### Taxonomy of Learning Problems: Supervised

Supervised Learning에서는 Training data가 모델이 예측해야하는 변수의 값을 포함한다.

- **Classification**: categorial variable
- **Regression**: continuous variable
- Ordinal regression, finite, ordered set of values
- **Rankings**: ordering of elements
- **Structured prediction**: sequence, tree, graph, …
- **Recommendation**: Item-by-user matrix

## Unsupervised Learning

Unsupervised learning에서 사용되는 Training data는, supervised 에서와는 달리, input, 즉 set of instances $x \in X$ 뿐이다. 여기에는 target variable이 없다. instances만이 주어졌기 때문에 우리가 직접 데이터의 형성 과정에 대한 추가적인 assumption을 만들어야 한다. 예를 들어 무작위 변수들의 독립성이 있겠다.

이 unsupervised learning의 목표는 데이터에서 structure의 detection이다. 예를 들어, 특정 properties를 공유하는 인스턴스들의 클러스트로의 가장 가능성있는, 그러나 데이터에서 아직은 관찰되지 않은 grouping을 찾는다. 

일반적으로 군집화(clustering), 차원 축소(dimensionality reduction), 이상치 탐지(anomaly detection) 등의 작업에 사용된다.

### Taxonomy of Learning Problems: Unsupervised

Unsupervised에서는 데이터들의 structural properties를 발견하는것을 목표로 삼는다.

- Clustering
- Unsupervised feature learning: 데이터를 잘 설명할 수 있는 attributes를 찾는다.


## Reinforcement Learning

Reinforcement Learning은 시스템을 제어하고 원하는 목표를 달성하기 위해 시행착오를 통해 학습하는 방법이다. 여기서는 시스템의 현재 **statement**를 관찰하고 행동을 선택하여 시스템에 영향을 준다. 이후 시스템의 반응을 통해 얻은 보상을 기반으로 학습이 이루어진다.

강화 학습의 목표는 주어진 환경에서 최적의 정책(policy)을 학습하는 것이다. Policy는 주어진 상태에서 취해야 할 행동을 결정하는 전략을 의미한다. Reinforcement learning은 주어진 환경에서 policy를 향상시키기 위해 exploration과 exploitation 사이에서 균형을 유지한다. 

* 환경(Environment)에 대한 사전지식이 없는 상태로 학습을 진행합니다.
* 보상을 통하여 학습을 합니다.
* 컴퓨터가 선택한 행동(Action)에 대한 환경의 반응에 따라 보상이 주어집니다.
* 행동의 결과로 나타나는 것이 보상이며 이것을 통하여 학습을 진행합니다.
* 보상을 최대한 많이 얻도록 하는 행동을 유도하도록 학습을 진행합니다.

### Learning to Control a System

Reinforcement Learning은 다음과 같은 system control을 학습하는 작업을 수행한다.

- contrl parameters를 가지는 시스템이 있다고 가정한다.
- utility function은 desired system behavior를 의미한다.
- control paramters의 변화는 시간이 지나도록 영향을 미칠 수 있다.
- learner는 desired behavior를 달성하는 모델을 찾기 위해 시스템을 실험하여야 한다. 이를 Exploration이라 한다.
- 동시에 시스템은 가능한 best state를 유지하여야 한다. 이를 Exploitation이라 한다.

Exploration과 Exploitation의 균형을 유지하는 것이 중요하다. 새로운 행동을 시도하고 새로운 정보를 얻기 위해 Exploration을 하면서, 동시에 현재의 최적 상태를 유지하기 위해 Exploitation을 해야 한다. 이를 통해 시스템을 효율적으로 제어하고, 원하는 동작을 달성할 수 있다.     
예를 들어, 자율 주행 자동차의 경우, Exploration을 통해 새로운 환경에 대한 정보를 수집하고, Exploitation을 통해 안전하게 목적지에 도착할 수 있는 최적의 경로를 선택한다.

### Example: Advertisement placement

광고를 배치하는데 있어 reinforcement learning은 사용자들의 클릭을 최대화하기 위해 광고를 배치하는 최적의 전략을 학습하는 것을 의미한다. 이 예시에서, 사용자가 어떤 광고를 클릭하는지를 학습하기 위해 learner는 실험을 해야한다. 그러나 가장 인기 있는 광고 이외의 광고를 사용할 때 학습자가 실험을 하는 경우에는 판매량이 감소할 수 있다.

### Taxonomy of Learning Problems: Reinforcement

다이나믹한 시스템을 컨트롤 하는 방법을 학습한다.


## Model or Parameter Space

Model space 혹은 parameter space $\Theta$는 다음과 같은 특징을 가진다:

- Model space는 모든 가능한 모델 또는 Classifier의 집합을 의미한다.
- Classifier는 Parameter $\theta \in \Theta$를 가진다.
- $\Theta$는 learning methode에 적합한 모델(classifier)의 집합이다.
- model space는 maschine learning을 위한 자유도(degree of freedom)중 하나이고, 흔히 사용되는 많은 spaces가 있다.
- Language Bias라고도 한다.

머신러닝, 혹은 딥러닝은 어떻게 보면 x에서 y로의 함수관계를 찾는 문제로 볼 수 있다. 이 때 함수의 구조와 복잡도를 정하는 것이 모델링 과정이다. 모델이 정해지면 모델의 parameter가 정해지게 되는데, 딥네트워크에서는 input과 output사이에 존재하는 모든 weight들이 모델의 parameter가 된다. model parameter는 model에 종속되어 정해진다.

모델과 모델 parameter가 정해지면 이제 남은 일은 모델 파라미터의 값을 정하는 것이다. 주어진 input(x)에 대해 원하는 output(y)이 나오도록 모델 parameter 값을 조정하는 것이 목적이다. 수집된 관측값들이 (xi; yi), i = 1, 2, ..., n라면 각각의 i에 대해 최대한 yi = fw(xi)가 되도록 parameter 값을 찾게 되는데, 이 과정이 딥러닝의 학습과정 또는 parameter 추정 과정에 해당한다.

그런데, 파라미터 추정을 위해서는 먼저 파라미터의 좋고 나쁨을 판단하는 기준이 있어야 한다. 그것이 바로 **loss function**를 정하는 일이다. 앞서 과정에서 **모델과 모델 파라미터가 정해지면 일종의 파라미터 공간(parameter space)이 생성**된다. 이 공간의 **차원은 모델 파라미터의 개수와 동일**하며 아직은 차원만 정해진 빈 공간이다. 그런데, loss function가 정해지면 이 공간에 산맥이 생기고 골짜기가 생긴다. 즉, Loss 함수에 따라서 각 파라미터 값에 대한 score(or 손실)가 매겨지기 때문에 이 score에 따라서 파라미터 공간에 높고 낮음이 생성된다.

딥러닝 학습에 있어서 마지막 선택은 주어진 산맥과 골짜기의 파라미터 공간에서 어떻게 최적의 답을 찾아갈 것인가이다. 파라미터 공간이 정해지면 이미 최적의 답도 같이 정해진다. 하지만 무한한 파라미터 공간을 미리 다 evaluation 해볼 수는 없기 때문에 현실적으로는 시작 위치에서 조금씩 해를 찾아갈 수 밖에 없다. 즉, gradient descent 방법, Gauss-Newton 방법, LM 방법 등과 같은 최적화 기법(optimizer)이 필요하다.
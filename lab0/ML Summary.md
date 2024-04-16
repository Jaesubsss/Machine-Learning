# Intelligent Data Analysis & Machine Learning

## Supervised Learning

### Classification
강의 1 초반부, 채워야함
### Regression

### Ordinal Regression

### Metric Learning

Metric Learning은 Classification 문제에서 새로운 클래스를 식별하는 것을 목표로 하는 방법 중 하나이다. 일반적으로 Classification problem에서는 training time에 이미 정의되고 고정된 classes가 있다.    
그러나 일부 model에서는 이전에 본 적 없는 클래스를 식별할 수 있어야 한다. 예를 들어, 생체 인식에서는 모든 개인의 얼굴 이미지, 지문 스캔, 홍채 스캔을 하나의 클래스로 볼 수 있다. 새로운 개인은 그에 따라 인식되어야 한다.

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


## Classifier Learning

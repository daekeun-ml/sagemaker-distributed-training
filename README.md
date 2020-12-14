# SageMaker Distributed Training
***본 문서는 [SageMaker 분산 훈련 공식 예제](https://github.com/aws/amazon-sagemaker-examples/blob/master/training/distributed_training/index.rst)의 한국어 버전입니다.***

SageMaker 분산 교육 라이브러리는 데이터 병렬(data-parallel) 및 모델 병렬(model-parallel) 훈련 전략을 모두 제공합니다. 소프트웨어 및 하드웨어 기술을 결합하여 GPU 간 및 노드 간 통신을 개선합니다. 훈련 스크립트를 조금만 수정하면 SageMaker의 분산 훈련 기능을 이용할 수 있습니다.

## SageMaker distributed data parallel
​
SageMaker 분산 데이터 병렬 (SDP)은 거의 선형적인 확장 효율성으로 딥러닝 모델에 대한 SageMaker의 훈련 기능을 확장하여 최소한의 코드 변경으로 빠른 훈련 시간을 달성합니다.

SDP는 AWS 네트워크 인프라 및 EC2 인스턴스 토폴로지에 대한 훈련 작업을 최적화합니다.

SDP는 gradient 업데이트를 활용하여 사용자 정의 AllReduce 알고리즘으로 노드간 통신을 수행합니다.

많은 양의 데이터로 모델을 훈련할 때 머신 러닝 실무자는 훈련 시간을 줄이기 위해 분산 훈련으로 전환하는 경우가 많습니다. 시간이 중요한 경우에 비즈니스 요구 사항은 가능한 한 빨리 또는 최소한 제한된 기간 내에 훈련을 완료하는 것입니다. 그런 다음 분산 훈련은 여러 노드의 클러스터들을 사용하도록 확장됩니다. 즉, 컴퓨팅 인스턴스의 여러 GPU들이 아니라 여러 GPU들이 있는 여러 인스턴스들을 의미합니다. 하지만, 클러스터 크기가 증가하면 성능이 크게 저하됩니다. 이러한 성능 저하는 주로 클러스터의 노드 간 통신 오버 헤드로 인해 발생합니다.

SageMaker 분산(SMD)은 분산 교육을 위해 SageMaker 모델 병렬(SMP) 및 SageMaker 데이터 병렬(SDP)의 두 가지 옵션을 제공합니다. 이 가이드는 데이터 병렬 전략을 사용하여 모델을 훈련하는 방법에 중점을 둡니다. 더 자세한 내용은 아래 링크를 참조해 주세요.

관련 링크 :
- [SageMaker data parallel developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html)
- [SageMaker Python SDK - data parallel APIs](https://sagemaker.readthedocs.io/en/stable/api/training/smd_data_parallel.html)

## SageMaker distributed data parallel
Amazon SageMaker 분산 모델 병렬 (SMP)은 이전에는 GPU 메모리 제한으로 인해 훈련하기 어려웠던 대규모 딥러닝 모델을 훈련하기 위한 모델 병렬 처리 라이브러리입니다. SMP는 여러 GPU 및 인스턴스에서 모델을 자동으로 효율적으로 분할하고 모델 훈련을 조정하므로 더 많은 매개 변수로 더 큰 모델을 생성하여 예측 정확도를 높일 수 있습니다.

SMP를 사용하여 최소한의 코드 변경으로 기존 TensorFlow 및 PyTorch 워크로드를 여러 GPU로 자동 분할할 수 있습니다. SMP API는 Amazon SageMaker SDK를 통해 액세스할 수 있습니다. 더 자세한 내용은 아래 링크를 참조해 주세요.

관련 링크:
- [SageMaker model parallel developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html)
- [SageMaker Python SDK - model parallel APIs](https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel.html)


## PyTorch 분산 훈련

### SageMaker distributed data parallel (SDP)
- [MNIST](pytorch/data_parallel/mnist/pytorch_smdataparallel_mnist_demo.ipynb)
- [MNIST Inference](pytorch/data_parallel/mnist/infer_pytorch.ipynb)
- [MaskRCNN](pytorch/data_parallel/maskrcnn/pytorch_smdataparallel_maskrcnn_demo.ipynb)
- [BERT](pytorch/data_parallel/bert/pytorch_smdataparallel_bert_demo.ipynb)

### SageMaker distributed model parallel (SMP)
- [MNIST](pytorch/model_parallel/mnist/pytorch_smmodelparallel_mnist.ipynb)
- [BERT](pytorch/model_parallel/bert/smp_bert_tutorial.ipynb)

## TensorFlow2 분산 훈련

### SageMaker distributed data parallel (SDP)
- [MNIST](tensorflow/data_parallel/mnist/tensorflow2_smdataparallel_mnist_demo.ipynb)
- [MNIST Inference](tensorflow/data_parallel/mnist/infer_tensorflow.ipynb)
- [MaskRCNN](tensorflow/data_parallel/maskrcnn/tensorflow2_smdataparallel_maskrcnn_demo.ipynb)
- [BERT](tensorflow/data_parallel/bert/tensorflow2_smdataparallel_bert_demo.ipynb)

### SageMaker distributed model parallel (SMP)
- [MNIST](tensorflow/model_parallel/mnist/tensorflow_smmodelparallel_mnist.ipynb)

## License Summary

이 샘플 코드는 MIT-0 라이센스에 따라 제공됩니다. LICENSE 파일을 참조하십시오.
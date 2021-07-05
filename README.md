# Discord 감성 채팅봇



<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/game.gif"></p>



디스코드 서버에서 사용자의 메세지에 나타나는 감정에 맞는 GIF Image를 보여주는 봇입니다.



<p align="center">tensorflow:2.5.0,	  TFX:0.30.0,	  	Kubeflow:1.7.1,	</p>



Data : [Big](https://aihub.or.kr/aidata/7978), [Small](https://github.com/songys/Chatbot_data)

*Discord Firebot project 21/05 ~ In progress* 

*moved from gitlab*

<br>

## Intro

처음에는 '*평소 게임을 하며 이용하던 디스코드 서버에 내가 만든 봇이 활동을 하면 어떨까*?' 로 만들기 시작해 

기능을 하나씩 덧붙여 나가다가 최근에 **머신러닝 서비스**를 봇에 도입하고 싶어서 시작하게 된 프로젝트 입니다.



단순히 NLP 모델을 학습시켜 사용자의 메세지에 적절한 GIF 이미지로 응답하는것을 넘어서서,  

**이상적인 ML Production 환경**을 구축하는것을 목표로 삼았습니다.



인터넷 채팅에서 나오는 엄청난 양의 live message data중 일부가 사람의 feed back 으로 labeling된다면, 실제 Production 환경에서 발생할 수 있는 data skew와 drift를 바로잡아줄 수 있는 매우 고품질의 Training example이 됩니다.



따라서 이와 같은 새로운 Dataset을 수집하고,  해당 Dataset을 통해 지속적으로 학습하고 배포할 수 있는 **CI/CD** 환경을 구축하기 위해 노력했습니다.



*(discord bot의 main function은 비공개 개발이므로 삭제했습니다. (감성 채팅봇이랑 관련없음))*

<br>

## Overall Structure

**사용자의 피드백으로 부터 새로운 감정 데이터쌍을 생성해 새로운 모델의 학습에 사용되는 전체 개요도**

<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/CICD.png"></p>

<br>

**사용자의 명령어 호출로부터 GIF를 받기 까지의 과정**

<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/structure.png"></p>

<br>

## Index

-> [Files](https://github.com/PyoJunCode/discord_chatbot#Files)

-> [Data Cleansing](https://github.com/PyoJunCode/discord_chatbot#Data-Cleansing)

-> [Models](https://github.com/PyoJunCode/discord_chatbot#Models)

-> [Serving](https://github.com/PyoJunCode/discord_chatbot#Serving)

-> [CI/CD](https://github.com/PyoJunCode/discord_chatbot#CI-CD)

->[Kubeflow Pipeline 구축 시 유의사항](https://github.com/PyoJunCode/discord_chatbot#Set-up-Kubeflow-cluster)

-> [Todo](https://github.com/PyoJunCode/discord_chatbot#Todo)

<br>

## Stacks

- tensorflow, 
- TFX,
- kubeflow, 
- discordbot api, 
- AWS Deep Learning AMI,
- AWS S3

<br>

## Files

- bot/

  discord bot 폴더입니다. (bot/cogs/chatbot.py 말고는 해당 프로젝트와 직접적인 연관은 없습니다.)

- data/

  Training에 사용했던 Data들입니다.

- images/

  Readme images

- saved_model/

  Pipeline의 pusher 결과물. 해당 폴더를 가지고 Inference server를 개방합니다.

- koELECTRA_pre_trained.ipynb

  Big dataset에 대한 Model의 TFX pipeline interative context notebook.

- simple_transformer.ipynb

  Small dataset에 대한 Model의 notebook

- tokenizer.subwords

  Small model의 subword tokenizer



## Data Cleansing



Data를 학습에 적합하게 만들기 위해서 몇가지 Cleansing/Preprocessing  과정을 거쳤습니다.



한글이 깨지는 문제를 해결하기 위해 utf-8 excel file -> euc-kr csv file 변환을 거쳤습니다.

### Anomalies

<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/anomalies.png"></p>

우선 TFX pipeline components인 **StatisticGen**을 통해 Training data에 불순물이 들어있는것을 확인하였습니다.

<br>

*compare train/val statistics*

<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/compare.png"></p>

<br>

*check anomalies*

<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/anomalies_found.png"></p>

ExampleValidator를 visualization하여 확실하게 anomalies임을 확인하고, 제외시켰습니다.



### Label



<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/label.png"></p>

csv의 label column Raw data는 '기쁨' '당황' '분노' '불안' '상처' '슬픔' 으로 표현되어 있습니다.

따라서 학습의 label로 사용하기 위해 sklearn의 Label Encoder를 사용하여 0 - 5의 값으로 변환하였습니다.



### Tokenize

* **koELECTRA** 모델의 경우 자체적으로 제공되는 35000 단어 규모의 Wordpiece tokenizer를 사용했습니다.

* **transformer** 모델의 경우 tfds에서 제공하는 SubwordTextEocder를 사용하였습니다.





<br>

## Models



### Big Model

colab, AI platform, Deep learning AMI 등의 비용 문제로 Pre-trained koELECTRA-small-v3 모델을 사용하였습니다.

 35000 len Wordpiece  tokenizer,

 512 Batch, 

512 max seq len

<br>

### Small Model

간단한 수준의 transformer encoder를 사용하였습니다.

20000 len Subword tokenizer

32 Batch,

8 Multi head attention

128 Position wide embedding

128 max seq len



<br>

## Serving



<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/serving.png"></p>



(Discordbot의 api와 관련된 부분은 설명을 생략하겠습니다.)

### Model Deploy

Model은 TFX pipeline의 Pusher에서 배포됩니다. 

Savedmodel 파일이 AWS Deep Learning AMI에 전송이 되면, 해당 인스턴스 안에서 작동되고 있는

Tensorflow Serving이 Model 파일을 통해 Inference를 시작해주는 포트를 개방합니다.



```
#Docker의 경우

docker run -t --rm -p 8501:8501 -v "/<saved model 파일의 경로>/saved_model:/models/<모델 이름>" -e MODEL_NAME=chatbot tensorflow/serving &

#예시
docker run -t --rm -p 8501:8501     -v "/$TESTDATA/saved_model:/models/chatbot"     -e MODEL_NAME=chatbot     tensorflow/serving &
```



### Get Inference



<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/sad.gif"></p>

 채팅창에 !반응 '할말' 을 치면 GIF 반응이 올라옵니다.



**cogs.chatbot.py**

```python
#Inference Server에 Predict request

parse = self.encoding(arg) #encoding 함수를 통해 문장을 tokenize 합니다.
        data = json.dumps({"signature_name": "serving_default", "instances":parse.tolist()}) # data 를 JSON으로 dump 합니다.

        #Change localhost to AWS serving server
        headers = {'content-type': 'application/json'}
        json_response = requests.post('http://localhost:8501/v1/models/chatbot:predict', data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions'] #예측 요청

        category = np.argmax(predictions) #결과 argmax
```

```python
#위에 나온 tokenize 함수
#tokenizer는 __init__ 에 정의되어있습니다.

    def encoding(self,sentence):

        sentence = self.preprocess_sentence(sentence)

        sentence = tf.expand_dims(
          self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)

        sentence = tf.keras.preprocessing.sequence.pad_sequences(
          sentence, maxlen=self.max_len, padding='post')

        return sentence



    def preprocess_sentence(self,sentence):
      sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
      sentence = sentence.strip()
      return sentence
```





<br>



## CI CD



<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/react.gif"></p>





앞서 설명했듯 Discord bot의 **잘못된 GIF 응답**에 대해서, 유저의 feedback으로 emoji를 응답받으면 새**로운 데이터쌍**이 생성됩니다.  (discordbot API로 tracking)

(ex: !반응 나 화났어 에 대해 잘못된 GIF가 나와서 유저가 :angry:로 reaction -> msg: '나 화났어', label: 'angry'​) 의 데이터 생성)

<br>



해당 과정이 Discord bot을 사용하는 모든 유저에 의해 반복되면, Discord bot이 위치한 EC2 서버에 새로운 데이터가 쌓이게 됩니다.



**cogs.chatbot.py**

```python
    @commands.command(name="업로드")
    async def upload(self,ctx):

        file = 'newdadta.pickle'
        name = 'newdata.pickle'

        s3 = boto3.client(
        's3',  # instance name
        aws_access_key_id="YOUR_ID",         # ID
        aws_secret_access_key="YOUR_KEY")    # Secret_key

        s3.upload_file(file,"BUCKET_NAME_YOU_WANT",name)
```



일정한 주기마다 디스코드 채팅창에 **!업로드** 를 입력하면, 이때까지 쌓은 새로운 데이터들이 연결된 AWS S3의 bucket에 저장됩니다.



<p align="center"><img src="https://github.com/PyoJunCode/discord_chatbot/blob/master/images/s3bucket.png"></p>

<br>

그후,

GCP의 AI platform notebook에서 해당 S3 bucket에 접근하여 새로운 데이터를 불러와 기존의 데이터와 합칩니다.

합치는 과정에서 TFDV(Tensorflow Data validation)을 이용해 Data의 skew, shift 등을 확인해 볼 수 있습니다.



그 다음은 [Pipeline 과정](https://github.com/PyoJunCode/data-centric-pipeline#Kubeflow-Pipeline)을 거쳐 Validation, Transform을 거쳐 Model의 Train 에 사용되어 **CI/CD의 한 주기**가 끝나게 됩니다.



봇의 사용자가 있는 한, 해당 과정의 반복을 통해 계속해서 데이터를 증강하고, 해당 데이터를 사용해 새로운 Model을 Training할 수 있습니다.



*(Pusher를 통해 A/B compare, Canary deploy 등의 작업을 할 수 있지만 여기서는 구현하지 않았습니다.)*





<br>

## Set up Kubeflow cluster

Kubeflow Pipeline을 구축하시려는 분들의 팁.



시작하기에 앞서, 현재 TFX, Kubeflow, AI Platform의 조합은 아직 과도기 단계이기 때문에 많은 Issue가 존재하고, 그만큼 빠른 버전 업데이트가 일어나고 있습니다.



하지만 현재 GCP AI platform이 공식적으로 지원하는 Kubeflow의 최신 버전은 1.4.1입니다.



Issue가 최대한 많이 고쳐진 버젼을 사용하기 위해 수동적으로 Kubeflow pipeline stand-alone 최신 버전을 설치하여 사용하는 것을 추천합니다.



글 작성 당시 최신버전은 1.7.0-alpha.2 이며, Google AI Platform의 Terminal을 열어 아래의 명령어를 차례대로 실행하면 됩니다.

<br>

```
CLUSTER_NAME="chatbot"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-2" # A machine with 2 CPUs and 7.50GB memory
SCOPES="cloud-platform" # This scope is needed for running some pipeline samples. Read the warning below for its security implication

gcloud container clusters create $CLUSTER_NAME \
     --zone $ZONE \
     --machine-type $MACHINE_TYPE \
     --scopes $SCOPES
```

<br>

```
export PIPELINE_VERSION=1.7.0-alpha.2
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
```





## Todo



* DVC를 통한 데이터 버져닝
* Data pipeline을 Airflow를 통해 구현 (Airflow on k8s)
* Model을 publising 할 때 A/B compare, Canary deployment 등을 적용
* S3 bucket에 데이터를 저장할 때 주기적으로 기존의 csv에 내용을 추가하여 업데이트하는 로직 적용

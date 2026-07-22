# -*- coding: utf-8 -*-
"""각 파일별 리포트 PDF 생성 (keras / ML) - 모바일 열람용."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Preformatted, Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))
SERIF = 'HYSMyeongJo-Medium'
SANS = 'HYGothic-Medium'

BASE = '/home/user/NSU_ALL'
OUT = os.path.join(BASE, 'reports')
os.makedirs(OUT, exist_ok=True)

# ---------- 스타일 ----------
styles = getSampleStyleSheet()
ACCENT = colors.HexColor('#2563eb')
DARK = colors.HexColor('#1e293b')
GREY = colors.HexColor('#475569')
LIGHT = colors.HexColor('#f1f5f9')
CODEBG = colors.HexColor('#f8fafc')

st_cover_title = ParagraphStyle('cov', fontName=SANS, fontSize=30, textColor=DARK, leading=38, spaceAfter=6)
st_cover_sub = ParagraphStyle('covs', fontName=SERIF, fontSize=13, textColor=GREY, leading=20)
st_h1 = ParagraphStyle('h1', fontName=SANS, fontSize=17, textColor=ACCENT, leading=22, spaceBefore=4, spaceAfter=8)
st_label = ParagraphStyle('lab', fontName=SANS, fontSize=10.5, textColor=colors.white, leading=14)
st_summary = ParagraphStyle('sum', fontName=SERIF, fontSize=11.5, textColor=DARK, leading=18, spaceAfter=4)
st_sec = ParagraphStyle('sec', fontName=SANS, fontSize=11.5, textColor=DARK, leading=16, spaceBefore=8, spaceAfter=3)
st_body = ParagraphStyle('body', fontName=SERIF, fontSize=10.5, textColor=colors.HexColor('#334155'), leading=16.5)
st_bullet = ParagraphStyle('bul', fontName=SERIF, fontSize=10.5, textColor=colors.HexColor('#334155'),
                           leading=16, leftIndent=12, bulletIndent=2, spaceAfter=1)
st_code = ParagraphStyle('code', fontName=SERIF, fontSize=7.6, textColor=colors.HexColor('#0f172a'),
                         leading=10.4, backColor=CODEBG, borderColor=colors.HexColor('#cbd5e1'),
                         borderWidth=0.5, borderPadding=7, leftIndent=2, rightIndent=2)
st_toc = ParagraphStyle('toc', fontName=SERIF, fontSize=10.5, textColor=DARK, leading=17)


def label_bar(text):
    """작은 색상 라벨 바."""
    t = Table([[Paragraph(text, st_label)]], colWidths=[None])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), ACCENT),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def code_block(code):
    # Preformatted는 페이지 경계에서 자동 분할됨 (긴 코드 대응)
    return Preformatted(code.rstrip('\n'), st_code)


def read_code(folder, fname):
    with open(os.path.join(BASE, folder, fname), encoding='utf-8') as f:
        return f.read()


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def build_report(folder, title_ko, subtitle, entries, out_name):
    doc = SimpleDocTemplate(os.path.join(OUT, out_name), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=title_ko)
    S = []
    # 표지
    S.append(Spacer(1, 60 * mm))
    S.append(Paragraph(title_ko, st_cover_title))
    S.append(HRFlowable(width='40%', thickness=3, color=ACCENT, spaceBefore=8, spaceAfter=14, hAlign='LEFT'))
    S.append(Paragraph(subtitle, st_cover_sub))
    S.append(Spacer(1, 10 * mm))
    S.append(Paragraph(f'총 {len(entries)}개 파일 · 파일별 리포트', st_cover_sub))
    S.append(PageBreak())

    # 목차
    S.append(Paragraph('목차', st_h1))
    S.append(HRFlowable(width='100%', thickness=0.7, color=colors.HexColor('#cbd5e1'), spaceAfter=8))
    toc_rows = []
    for i, e in enumerate(entries, 1):
        toc_rows.append([Paragraph(f'{i:02d}', st_toc),
                         Paragraph(f'<b>{esc(e["file"])}</b>', st_toc),
                         Paragraph(e['one'], st_toc)])
    tt = Table(toc_rows, colWidths=[10 * mm, 55 * mm, 105 * mm])
    tt.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LINEBELOW', (0, 0), (-1, -2), 0.3, colors.HexColor('#e2e8f0')),
        ('TEXTCOLOR', (0, 0), (0, -1), ACCENT),
    ]))
    S.append(tt)
    S.append(PageBreak())

    # 파일별 리포트
    for i, e in enumerate(entries, 1):
        S.append(label_bar(f'파일 {i:02d} / {len(entries)}'))
        S.append(Spacer(1, 4))
        S.append(Paragraph(esc(e['file']), st_h1))
        S.append(HRFlowable(width='100%', thickness=0.7, color=colors.HexColor('#cbd5e1'), spaceAfter=6))

        S.append(Paragraph('한 줄 요약', st_sec))
        S.append(Paragraph(e['summary'], st_summary))

        S.append(Paragraph('핵심 개념', st_sec))
        for b in e['concepts']:
            S.append(Paragraph('• ' + b, st_bullet))

        if e.get('flow'):
            S.append(Paragraph('코드 흐름', st_sec))
            S.append(Paragraph(e['flow'], st_body))

        S.append(Paragraph('전체 코드', st_sec))
        S.append(code_block(read_code(folder, e['file'])))
        S.append(PageBreak())

    doc.build(S)
    print('생성:', os.path.join(OUT, out_name))


# ==================================================================
# keras 리포트 데이터
# ==================================================================
keras_entries = [
 {'file':'keras01.py','one':'가장 단순한 선형회귀 첫 예제',
  'summary':'단층 Dense(1) 신경망으로 y=x 관계를 학습시켜 딥러닝의 기본 4단계(데이터→모델→컴파일/훈련→예측)를 처음 체험하는 파일.',
  'concepts':['Sequential 모델과 Dense(1, input_dim=1) 단층 구성','loss=mse, optimizer=adam 로 회귀 학습','model.fit(epochs)로 반복 훈련, model.predict로 예측','x=y=[1,2,3] 이므로 4 입력 시 4에 가까운 값 예측'],
  'flow':'데이터 정의 → 단층 모델 생성 → mse/adam 컴파일 → 2000 에폭 훈련 → 4의 예측값 출력. 신경망이 w·x+b의 w≈1, b≈0을 찾아가는 과정을 관찰한다.'},
 {'file':'keras02.py','one':'선형회귀 데이터 확장 버전',
  'summary':'keras01과 동일 구조에서 데이터를 1~6으로 늘리고 7을 예측. 데이터 양 증가와 에폭 수(2300)의 영향을 체감하는 파일.',
  'concepts':['y=wx+b 형태에서 w,b가 랜덤 초기값으로 시작','단층 Dense(1) 유지','더 많은 데이터·에폭으로 학습 안정성 확인'],
  'flow':'1~6 데이터로 훈련 후 7을 예측하여 7에 가까운 값이 나오는지 확인.'},
 {'file':'keras03_deep1.py','one':'다층(딥) 신경망 첫 구성',
  'summary':'여러 Dense 층을 쌓아 처음으로 깊은 신경망(DNN)을 만든다. 각 층마다 input_dim을 명시한 장황한 표기 버전.',
  'concepts':['은닉층을 여러 개 쌓아 표현력 향상','각 층 output dim이 다음 층 input이 됨','층마다 input_dim 명시(초보 이해용, 실제로는 불필요)','epochs=100로 단축'],
  'flow':'8→10→8→6→4→3→1 형태로 층을 쌓고 7을 예측. 깊이가 늘어도 단순 데이터라 성능 차이는 크지 않음.'},
 {'file':'keras03_deep2.py','one':'딥 신경망 간결 표기',
  'summary':'deep1과 동일하지만 두 번째 층부터 input_dim을 생략한 간결한 표기법을 사용. 실무에서 쓰는 방식.',
  'concepts':['첫 층만 입력 차원 지정하면 나머지는 자동 추론','Dense(뉴런수)만 써도 되는 이유','mse=평균제곱오차 개념 주석'],
  'flow':'12→10→7→6→4→3→1 구조로 간결하게 작성 후 예측.'},
 {'file':'keras04_evaluate.py','one':'evaluate로 성능 평가 추가',
  'summary':'훈련 후 model.evaluate로 loss를 측정하는 4단계(평가)를 추가. 예측 전에 모델 성능을 수치로 확인.',
  'concepts':['model.evaluate(x,y)가 최종 loss(mse) 반환','예측값과 실제값의 오차를 수치화','훈련 loss와 평가 loss의 의미 구분 시작'],
  'flow':'훈련 → evaluate로 loss 출력 → 7 예측. 평가 단계의 필요성을 학습.'},
 {'file':'keras05_batch.py','one':'batch_size 개념 도입',
  'summary':'fit에 batch_size=2를 추가하여 데이터를 그룹으로 잘라 계산하는 배치 학습을 소개.',
  'concepts':['batch_size: 한 번에 처리하는 데이터 묶음 크기','데이터가 많을 때 메모리·속도 관리','배치가 작을수록 갱신 잦음 / 클수록 안정적'],
  'flow':'batch_size=2로 훈련하여 배치 단위 갱신을 경험.'},
 {'file':'keras06_행렬.py','one':'넘파이 차원/shape 이해',
  'summary':'스칼라·벡터·행렬·텐서의 차원 개념과 numpy array의 .shape 읽는 법을 정리한 이론 파일.',
  'concepts':['0차원 스칼라, 1차원 벡터, 2차원 행렬, 3차원 텐서','.shape로 (행,열) 등 형태 확인','괄호 depth로 차원 판단하는 법','딥러닝 입력 shape 이해의 기초'],
  'flow':'다양한 array를 만들고 shape를 출력해 차원 규칙을 체득.'},
 {'file':'keras07_mlp1.py','one':'다중 입력(MLP) - 열 2개',
  'summary':'입력 특성(열)이 2개인 데이터를 다루는 MLP. input_shape=(2,)로 다차원 입력을 처음 경험.',
  'concepts':['MLP = Multi Layer Perceptron','열=컬럼=피쳐=특성=속성 용어 정리','input_shape=(2,)와 input_dim=2의 관계','데이터 (6,2) 형태로 열 개수 맞추기'],
  'flow':'(6,2) 입력 데이터로 훈련 후 [[7,13]] 예측. y가 1열이므로 출력은 1.'},
 {'file':'keras07_mlp2.py','one':'transpose로 데이터 형태 변환',
  'summary':'(2,6) 데이터를 .transpose()로 (6,2)로 바꿔 모델에 맞추는 실전 전처리 기법.',
  'concepts':['x.T / x.transpose()로 행렬 전치','대체로 input_dim보다 input_shape 사용','데이터 형태를 모델 입력에 맞추는 노가다 자동화'],
  'flow':'전치로 형태 정리 → 훈련 → 예측. 전처리의 중요성 인식.'},
 {'file':'keras07_mlp3.py','one':'입력 특성 3개 MLP',
  'summary':'3개 열을 가진 데이터를 input_shape=(3,)로 학습. 여러 특성을 동시에 입력하는 연습.',
  'concepts':['input_shape=(3,)는 (None,3) 의미, 행(샘플수)은 무관','.T로 (10,3) 형태 정리','예측 시에도 입력 형태를 맞춰야 함'],
  'flow':'3특성 데이터를 훈련하고 전치한 예측 입력으로 결과 확인.'},
 {'file':'keras07_mlp4.py','one':'출력 2개(다중 출력) 모델',
  'summary':'마지막 층을 Dense(2)로 만들어 출력을 2개로 내는 다중 출력 모델. 과적합 경고 포함.',
  'concepts':['출력층 뉴런 수 = 예측 대상(y) 열 수','다중 출력은 성능 저하 가능 → 보통 모델 분리','전체 데이터로 훈련 시 과적합(오버핏) 위험','train/test 분리 필요성 예고'],
  'flow':'입력3·출력2 모델 훈련 후 예측. 오버핏과 데이터 분리 개념으로 연결.'},
 {'file':'keras08_train_test1.py','one':'train_test_split로 데이터 분리',
  'summary':'sklearn의 train_test_split로 훈련/테스트를 7:3 랜덤 분리. 과적합을 막는 핵심 기법 도입.',
  'concepts':['train_test_split(test_size=0.3, random_state)','훈련 데이터로 학습, 테스트 데이터로 평가','random_state로 재현성 확보','평가는 반드시 안 본 데이터로'],
  'flow':'데이터 분리 → 훈련셋 학습 → 테스트셋 evaluate → 11 예측.'},
 {'file':'keras08_train_test2.py','one':'슬라이싱으로 수동 분리',
  'summary':'넘파이 슬라이싱(x[0:7], x[7:])으로 데이터를 직접 7:3 자르는 방법. split 함수와 비교 학습.',
  'concepts':['리스트 슬라이싱 x[start:end] 규칙','수동 분리는 순서대로만 가능(섞이지 않음)','exit()로 중간 확인 후 종료','자동 split의 장점 대비'],
  'flow':'슬라이싱으로 분리 확인(exit로 조기 종료). 수동 방식의 한계 인식.'},
 {'file':'keras08_train_test3.py','one':'shuffle·train_size 옵션 심화',
  'summary':'train_test_split의 shuffle, train_size, random_state 옵션을 자세히 다루는 버전.',
  'concepts':['train_size=0.7과 test_size 상호 보완','shuffle=True/False 차이','random_state 고정으로 결과 재현'],
  'flow':'섞어서 7:3 분리 → 훈련·평가 → 예측.'},
 {'file':'keras09_scatter1.py','one':'matplotlib 산점도 시각화',
  'summary':'예측 결과를 산점도(scatter)와 회귀선(plot)으로 그려 모델이 데이터를 얼마나 잘 맞추는지 시각화.',
  'concepts':['plt.scatter로 실제 데이터 점 찍기','plt.plot으로 예측 회귀선 그리기','노이즈가 있는 y 데이터로 실전 감각','에폭이 적으면(8) 선이 덜 맞음'],
  'flow':'분리·훈련·예측 후 그래프로 적합도 확인.'},
 {'file':'keras09_scatter2.py','one':'산점도 - 데이터 20개 확장',
  'summary':'데이터를 20개로 늘리고 층을 키워(epochs=300) 더 흩어진 데이터에 회귀선을 맞추는 연습.',
  'concepts':['데이터·에폭 증가로 적합도 향상','넓은 은닉층(100→50→60→40→1)','시각적으로 회귀선 품질 비교'],
  'flow':'20개 데이터 훈련 후 회귀선 시각화.'},
 {'file':'keras10_R2_RMSE.py','one':'R²·RMSE·MSE 평가지표',
  'summary':'회귀 성능을 loss뿐 아니라 R2 score, RMSE, MSE 등 표준 지표로 평가하는 법을 배운다.',
  'concepts':['r2_score: 1에 가까울수록 좋음','RMSE: mse에 루트, 원래 단위로 해석','mse=오차=cost, mae=절댓값 오차','sklearn.metrics 활용'],
  'flow':'예측 후 R2/RMSE/MSE를 각각 계산·출력하여 다각도 평가.'},
 {'file':'keras11_1_california.py','one':'캘리포니아 집값 회귀(실데이터)',
  'summary':'sklearn의 캘리포니아 주택 데이터(특성 8개)로 실전 회귀. 보스턴 데이터 대체용 교육 데이터셋.',
  'concepts':['fetch_california_housing, feature_names 8개','input_shape=(8,) 다특성 회귀','train_size·epochs·batch_size 튜닝 기록(주석)','R2 약 0.52 수준 도달'],
  'flow':'데이터 로드 → 분리 → 다층 회귀 훈련 → R2/RMSE 평가. 하단 주석에 하이퍼파라미터 실험 기록.'},
 {'file':'keras11_2_diabetes.py','one':'당뇨병 진행도 회귀',
  'summary':'load_diabetes(특성 10개)로 회귀. 주의: 중간에 Dense(0)이 들어간 오류가 있는 학습 중 코드.',
  'concepts':['load_diabetes 데이터 (442,10)','input_shape=(10,) 회귀','batch_size=4 소형 배치','⚠ Dense(0) 층은 실행 오류를 유발(디버깅 포인트)'],
  'flow':'분리·훈련·R2/RMSE 평가. Dense(0)은 실수로 남은 부분으로 실제로는 제거해야 함.'},
 {'file':'keras11_3_kaggle_bike.py','one':'캐글 자전거 수요 예측',
  'summary':'캐글 Bike Sharing CSV를 pandas로 읽어 회귀. 실제 대회 데이터 전처리의 시작.',
  'concepts':['pd.read_csv(index_col=0)로 날짜열 인덱스화','drop으로 casual/registered/count 제거','count를 타깃(y)으로 설정','time으로 훈련 시간 측정, RMSE 평가'],
  'flow':'CSV 로드 → 특성/타깃 분리 → 훈련(시간측정) → RMSE 평가.'},
 {'file':'keras12_verbose.py','one':'verbose 옵션과 학습 로그',
  'summary':'fit의 verbose 값(0~4)에 따라 학습 진행 표시가 달라지고 속도에도 영향을 준다는 것을 실험.',
  'concepts':['verbose=1 진행바, 2 텍스트, 3 에폭만, 0 무출력','출력 생략 시 딜레이 감소로 약간 빨라짐','DESCR/feature_names로 데이터 탐색','R2/RMSE/MSE 종합 평가'],
  'flow':'당뇨 데이터로 verbose를 바꿔가며 훈련 시간과 로그를 비교.'},
 {'file':'keras13_validation.py','one':'검증 데이터(validation_split)',
  'summary':'훈련 중 validation_split으로 검증셋을 떼어 과적합 여부를 실시간 감시. relu 활성화도 도입.',
  'concepts':['validation_split=0.2로 훈련 중 검증','val_loss가 신뢰도 높은 지표','activation=relu(음수 제거) vs linear(기본)','이상치/결측치 개념 주석, 제출 CSV 생성'],
  'flow':'CSV 로드 → relu 다층 → 검증 포함 훈련 → RMSE 평가 → 제출 파일 저장.'},
 {'file':'keras14_EarlyStopping.py','one':'EarlyStopping 콜백',
  'summary':'val_loss가 개선되지 않으면 자동으로 훈련을 멈추는 EarlyStopping으로 최적 시점을 찾는다.',
  'concepts':['monitor=val_loss, mode=min, patience','restore_best_weights=True로 최고 가중치 복원','epochs를 매우 크게 줘도 알아서 멈춤','로컬/글로벌 미니마 개념'],
  'flow':'콜백 등록 → 과도한 에폭으로 fit → 자동 정지 → 평가·제출.'},
 {'file':'keras14_EarlyStopping2_diabetes.py','one':'EarlyStopping - 당뇨 데이터',
  'summary':'당뇨 회귀에 EarlyStopping을 적용해 최적 지점에서 자동 정지시키는 연습.',
  'concepts':['load_diabetes에 EarlyStopping 적용','mode=auto/min 차이','R2/RMSE/MSE로 최종 평가'],
  'flow':'콜백과 함께 훈련 후 자동 정지, 지표 평가.'},
 {'file':'keras15_warning.py','one':'경고 메시지 숨기기',
  'summary':'warnings.filterwarnings("ignore")로 불필요한 경고 출력을 끄는 한 줄짜리 유틸 스니펫.',
  'concepts':['불필요한 warning 억제로 로그 깔끔하게','다른 스크립트 상단에 붙여 사용'],
  'flow':'경고 무시 설정 두 줄.'},
 {'file':'keras16)sigmoid_matris_cancer.py','one':'이진분류(유방암) sigmoid',
  'summary':'유방암 데이터로 이진분류. 출력층 sigmoid + binary_crossentropy + accuracy로 분류 문제 첫 도입.',
  'concepts':['이진분류=sigmoid+binary_crossentropy 고정','relu는 마지막 층에 쓰면 안 됨(0 이상만)','metrics=[accuracy]로 정확도 측정','np.round로 0/1 변환 후 accuracy_score'],
  'flow':'데이터 분리 → relu 은닉+sigmoid 출력 → 이진 훈련 → 정확도 평가.'},
 {'file':'keras17_sofrmax_OneHat_iris.py','one':'다중분류(붓꽃) softmax',
  'summary':'붓꽃 3종 다중분류. 원-핫 인코딩(get_dummies)+softmax+categorical_crossentropy 조합.',
  'concepts':['다중분류=softmax+categorical_crossentropy','pd.get_dummies로 원-핫 인코딩','출력층 Dense(3)=클래스 수','argmax로 클래스 복원 후 정확도'],
  'flow':'붓꽃 로드 → 원핫 → 다층+softmax 훈련 → argmax 예측 → 정확도.'},
 {'file':'keras18_softmax_wine.py','one':'다중분류(와인) softmax',
  'summary':'와인 3종(특성 13개) 다중분류. iris와 동일한 softmax 패턴을 다른 데이터에 적용.',
  'concepts':['특성 13개 input_shape=(13,)','클래스 3개 softmax 출력','EarlyStopping 병행','argmax 정확도 평가'],
  'flow':'와인 로드 → 원핫 → 훈련 → 정확도 평가.'},
 {'file':'keras19_gpu.test01.py','one':'GPU 인식 확인',
  'summary':'텐서플로가 GPU를 인식하는지 list_physical_devices로 점검하는 진단 스크립트.',
  'concepts':['tf.config.experimental.list_physical_devices("GPU")','GPU 유무 분기 출력','학습 가속 환경 확인'],
  'flow':'GPU 목록을 조회해 "돈다/없다" 출력.'},
 {'file':'keras20_load_digits.py','one':'손글씨 숫자 다중분류',
  'summary':'load_digits(8x8=64특성) 0~9 숫자 10-클래스 분류. softmax 출력 10개.',
  'concepts':['특성 64개, 클래스 10개','get_dummies 원핫 인코딩','출력층 Dense(10, softmax)','argmax로 예측 클래스, 정확도 평가'],
  'flow':'digits 로드 → 원핫 → 훈련 → 정확도. (중간 exit로 탐색 단계 구분)'},
 {'file':'keras21_summary.py','one':'model.summary()로 파라미터 확인',
  'summary':'model.summary()로 각 층의 출력 shape와 파라미터(가중치) 개수를 출력해 모델 구조를 이해.',
  'concepts':['파라미터 수 = (입력+1)×출력 (bias 포함)','층별 Output Shape 읽기','Total params로 모델 규모 파악'],
  'flow':'간단한 모델을 만들고 summary 출력. 주석에 실제 표 예시 첨부.'},
 {'file':'keras22_1_save_model.py','one':'모델 저장 - save()',
  'summary':'model.save(".keras")로 구성한 모델을 파일로 저장하는 법. 훈련 전 구조 저장 예시.',
  'concepts':['model.save(경로.keras) 상대경로','.keras 저장 포맷','저장/불러오기로 재사용','R2/RMSE/MSE 평가'],
  'flow':'모델 구성 → save → 훈련 → 평가.'},
 {'file':'keras22_2_load_model.py','one':'모델 불러오기 - load_model',
  'summary':'load_model로 저장된 모델을 불러와 사용. 저장한 모델이 훈련 결과를 덮어쓰는 흐름 관찰.',
  'concepts':['load_model(경로)로 복원','불러온 모델로 이어서 평가','save/load 위치에 따른 동작 차이'],
  'flow':'모델 구성·훈련 후 load_model로 교체하여 평가.'},
 {'file':'keras22_3_save_model2.py','one':'훈련 후 모델 저장',
  'summary':'훈련이 끝난 뒤 model.save로 가중치까지 포함해 저장하는 실전 패턴.',
  'concepts':['훈련 완료 후 저장 → 가중치 보존','keras_22_3.save_model.keras 생성','평가지표 확인'],
  'flow':'구성 → 훈련 → save → 평가.'},
 {'file':'keras22_4_load_model2.py','one':'훈련된 모델 불러와 재사용',
  'summary':'22_3에서 저장한 학습된 모델을 load_model로 불러와 바로 활용. 모델 배포 개념.',
  'concepts':['학습 완료 모델 로드로 재훈련 불필요','summary로 복원된 구조 확인','저장·배포 워크플로 완성'],
  'flow':'load_model → (재)훈련/요약 → 평가.'},
 {'file':'keras23_model_CheckPoint1.py','one':'ModelCheckpoint로 최적 저장',
  'summary':'EarlyStopping+ModelCheckpoint를 함께 써서 훈련 중 val_loss가 가장 좋을 때의 모델을 자동 저장.',
  'concepts':['ModelCheckpoint(save_best_only=True)','val_loss 최소 시점 자동 저장','EarlyStopping과 콜백 리스트 결합','제출 CSV 생성까지'],
  'flow':'콜백 2개 등록 → 훈련 → 최적 모델 파일 저장 → 평가·제출.'},
 {'file':'keras23_Model_CheckPoint2.py','one':'체크포인트 모델 불러와 평가',
  'summary':'체크포인트로 저장된 최적 모델(keras23_mcp1.keras)을 불러와 곧바로 평가. 모델 구성 코드 없이 재사용.',
  'concepts':['load_model로 최적 체크포인트 복원','모델 구성/훈련 생략하고 평가만','저장된 최적 가중치의 재현성'],
  'flow':'데이터 준비 → load_model → evaluate/predict.'},
 {'file':'keras24_Dropout.py','one':'Dropout 규제(과적합 방지)',
  'summary':'층 사이에 Dropout(0.3)을 넣어 일부 뉴런을 무작위로 꺼 과적합을 억제. evaluate에는 미적용.',
  'concepts':['Dropout: 훈련 중 뉴런 일부 랜덤 비활성','과적합 완화 규제 기법','evaluate/predict 시엔 Dropout 꺼짐','EarlyStopping+Checkpoint 병행'],
  'flow':'Dropout 삽입한 모델을 자전거 데이터로 훈련 → RMSE 평가. 드롭아웃 유무 loss 비교 주석.'},
 {'file':'keras25_MinMaxScaler_diavbet.py','one':'MinMaxScaler 정규화',
  'summary':'MinMaxScaler로 특성을 0~1로 정규화. train에 fit, test엔 transform만 적용하는 올바른 순서 학습.',
  'concepts':['fit은 x_train에만(데이터 누수 방지)','transform으로 train/test 동일 스케일','0~1 정규화로 학습 안정화','min/max 출력으로 확인'],
  'flow':'분리 → 스케일러 fit/transform → 훈련 → R2/RMSE/MSE.'},
 {'file':'keras26_StandardScaler.py','one':'StandardScaler 표준화',
  'summary':'평균0·분산1로 표준화. MinMax와 loss를 비교(주석)하며 스케일링 기법 선택 감각을 기른다.',
  'concepts':['fit_transform으로 간결하게','표준화 vs 정규화 결과 비교','스케일링이 회귀 성능에 미치는 영향'],
  'flow':'스케일링 → 훈련 → R2 평가. 주석에 stand/minmax loss 비교.'},
 {'file':'keras27_RobustScalerScaler.py','one':'RobustScaler(이상치 강건)',
  'summary':'중앙값·사분위수 기반 RobustScaler로 이상치에 강건한 스케일링을 적용.',
  'concepts':['RobustScaler는 이상치 영향 최소화','median/IQR 기반 변환','세 가지 스케일러(MinMax/Standard/Robust) 비교 완성'],
  'flow':'RobustScaler 적용 → 훈련 → R2 평가.'},
]

# ==================================================================
# ML 리포트 데이터
# ==================================================================
ml_entries = [
 {'file':'m01_and.py','one':'AND 게이트 - LinearSVC',
  'summary':'AND 논리연산을 sklearn LinearSVC로 학습. 선형 분류기로 완벽히 분리 가능한 첫 예제.',
  'concepts':['머신러닝은 역전파 없이 fit 한 번으로 학습(epochs 불필요)','model.score와 accuracy_score로 정확도','AND는 선형 분리 가능 → 정확도 1.0','LinearSVC 기본 사용법'],
  'flow':'AND 데이터 정의 → LinearSVC fit → 예측 → 정확도 1.0 확인.'},
 {'file':'m02_or.py','one':'OR 게이트 - LinearSVC',
  'summary':'OR 논리연산도 선형 분리 가능함을 LinearSVC로 확인. m01과 짝을 이루는 예제.',
  'concepts':['OR도 선형 분리 가능 → 정확도 1.0','y_data만 [0,1,1,1]로 변경','선형 모델의 적용 범위 이해'],
  'flow':'OR 데이터로 fit·예측·정확도 확인.'},
 {'file':'m03_xor_1.py','one':'XOR 한계 - Perceptron',
  'summary':'XOR는 선형 분리가 불가능해 단층 Perceptron이 실패함을 보여주는 핵심 예제(딥러닝 필요성의 근거).',
  'concepts':['XOR는 선형 분리 불가능','단층 퍼셉트론의 한계 → 정확도 낮음','다층 신경망이 필요한 이유','퍼셉트론 역사적 맥락'],
  'flow':'XOR 데이터에 Perceptron 적용 → 낮은 정확도로 한계 확인.'},
 {'file':'m03_xor_2_keras.py','one':'XOR - 단층 keras도 실패',
  'summary':'keras 단층 Dense(1,sigmoid)로 XOR을 시도해도 여전히 못 푸는 것을 확인. 단층의 본질적 한계.',
  'concepts':['단층 sigmoid로도 XOR 해결 불가','binary_crossentropy 사용','예측값이 확률이라 accuracy_score 주의','다층 필요성 강조'],
  'flow':'단층 keras로 XOR 훈련 → 낮은 정확도.'},
 {'file':'m03_xor_3_keras.py','one':'XOR 해결 - 다층 퍼셉트론',
  'summary':'은닉층(16→4)을 추가한 다층 신경망으로 마침내 XOR을 해결. 딥러닝의 힘을 증명하는 결정적 예제.',
  'concepts':['은닉층 추가로 비선형 문제 해결','relu 은닉 + sigmoid 출력','단층→다층 전환의 의미','MLP가 XOR을 학습 가능'],
  'flow':'은닉층 포함 모델로 XOR 훈련 → 해결 확인.'},
 {'file':'m04_01_california.py','one':'회귀 모델 비교 - 캘리포니아',
  'summary':'캘리포니아 집값에 여러 sklearn 모델을 바꿔 끼우며 회귀/분류 모델 구분과 성능 차이를 체감.',
  'concepts':['분류모델(LinearSVC/LogisticRegression)은 회귀에 부적합','DecisionTreeRegressor/RandomForestRegressor 사용','model.score로 R2 비교','주석에 각 모델 점수 기록'],
  'flow':'모델을 주석으로 바꿔가며 fit·score 비교.'},
 {'file':'m04_02_diabetes.py','one':'회귀 모델 비교 - 당뇨',
  'summary':'당뇨 데이터에 동일하게 여러 모델을 적용. DecisionTree가 1.0(과적합), RandomForest가 현실적 성능.',
  'concepts':['DecisionTree score 1.0 = 훈련데이터 과적합','RandomForest의 일반화 성능','회귀 전용 모델 선택','모델별 점수 주석 비교'],
  'flow':'모델 교체하며 score 비교, 과적합 관찰.'},
 {'file':'m05_for_1.py','one':'for문·enumerate 기초',
  'summary':'리스트 순회와 enumerate(인덱스+값) 사용법을 익히는 파이썬 기본 문법 파일.',
  'concepts':['for문으로 리스트 순회','enumerate(start=1)로 인덱스 함께','다음 파일의 반복 실험 준비'],
  'flow':'리스트를 순회하며 값과 인덱스를 출력.'},
 {'file':'m05_for_2_sample.py','one':'for문으로 데이터×모델 자동 비교',
  'summary':'여러 데이터셋과 여러 모델을 이중 for문으로 조합해 한 번에 성능을 비교하는 실전 자동화 패턴.',
  'concepts':['data_list × model_list 이중 반복(12회)','iris/cancer/wine + 4개 분류기','반복문으로 실험 자동화','model_name_list로 결과 라벨링'],
  'flow':'데이터별로 4개 모델을 자동 훈련·score 출력.'},
 {'file':'m06_all_estimators1_분류.py','one':'all_estimators - 전체 분류기 탐색',
  'summary':'sklearn의 모든 분류 모델을 all_estimators로 가져와 자동으로 훈련·평가하고 최고 모델을 찾는다.',
  'concepts':['all_estimators(type_filter=classifier)','try/except로 에러 모델 건너뛰기','RobustScaler 전처리','최고 정확도 모델 자동 선정'],
  'flow':'유방암 데이터로 전체 분류기 순회 → 최고모델 출력.'},
 {'file':'m06_all_estimators2_회귀.py','one':'all_estimators - 전체 회귀기 탐색',
  'summary':'모든 회귀 모델을 자동 순회하며 R2를 비교, 당뇨 데이터에서 최고 회귀 모델을 탐색.',
  'concepts':['type_filter=regressor로 회귀기 전체','예외 처리로 안정적 반복','최고 R2 모델 자동 추적','모델 탐색 자동화 완성'],
  'flow':'당뇨 데이터로 전체 회귀기 순회 → 최고모델 출력.'},
 {'file':'m07_kfoid01_iris.py','one':'K-Fold 교차검증',
  'summary':'KFold 교차검증(cross_val_score)으로 데이터를 5조각 내어 번갈아 검증, 더 신뢰도 높은 성능을 측정.',
  'concepts':['KFold(n_splits=5, shuffle) 교차검증','cross_val_score로 5회 점수','평균 정확도로 안정적 평가','n_jobs=-1 병렬 처리'],
  'flow':'붓꽃에 DecisionTree로 5-fold 교차검증 → 점수·평균 출력.'},
]

build_report('keras', 'Keras 딥러닝 학습 리포트',
             '텐서플로/케라스 기초부터 회귀·분류·저장·규제·스케일링까지 단계별 정리',
             keras_entries, 'keras_리포트.pdf')
build_report('ML', 'Machine Learning 학습 리포트',
             '퍼셉트론과 XOR 문제부터 sklearn 모델 비교·교차검증까지 정리',
             ml_entries, 'ML_리포트.pdf')
print('완료')

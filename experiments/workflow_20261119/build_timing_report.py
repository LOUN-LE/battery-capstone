from pathlib import Path
from datetime import datetime
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'work/aws1000_retrieved'
OUT=ROOT/'outputs/aws1000_delivery_20261116'
OUT.mkdir(parents=True,exist_ok=True)
report=json.loads((SRC/'dataset_report.json').read_text(encoding='utf-8-sig'))
started=json.loads((SRC/'started.json').read_text(encoding='utf-8-sig'))
done=json.loads((SRC/'completion.json').read_text(encoding='utf-8-sig'))
receipt=json.loads((SRC/'retrieval_receipt.json').read_text(encoding='utf-8-sig'))
elapsed=report['wall_seconds']
receipt_delta=(datetime.fromisoformat(done['finished_utc'])-datetime.fromisoformat(started['start_utc'])).total_seconds()
assert report['status_counts']=={'completed':1000,'censored':0,'failed':0,'pending':0}
assert receipt['instance_state']=='stopped'
pdfmetrics.registerFont(TTFont('Malgun',r'C:\Windows\Fonts\malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold',r'C:\Windows\Fonts\malgunbd.ttf'))
pdfmetrics.registerFontFamily('Malgun',normal='Malgun',bold='MalgunBold',italic='Malgun',boldItalic='MalgunBold')
ST={
 'title':ParagraphStyle('title',fontName='MalgunBold',fontSize=20,leading=28,textColor=colors.black,spaceAfter=12),
 'h':ParagraphStyle('h',fontName='MalgunBold',fontSize=12.5,leading=18,textColor=colors.black,spaceBefore=12,spaceAfter=7),
 'body':ParagraphStyle('body',fontName='Malgun',fontSize=10.1,leading=15.4,spaceAfter=7,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='Malgun',fontSize=8.8,leading=12.7,spaceAfter=5,wordWrap='CJK',textColor=colors.HexColor('#444444')),
 'cell':ParagraphStyle('cell',fontName='Malgun',fontSize=9.1,leading=13.2,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='MalgunBold',fontSize=9.1,leading=13.2,wordWrap='CJK',textColor=colors.white),
}
story=[]
text_sections=[]
def para(t,kind='body'):
    story.append(Paragraph(t,ST[kind]));text_sections.append(t.replace('<b>','').replace('</b>','').replace('<br/>','\n'))
def table(rows,widths):
    cells=[[Paragraph(escape(str(v)).replace('\n','<br/>'),ST['head' if i==0 else 'cell']) for v in row] for i,row in enumerate(rows)]
    tb=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tb.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#233E56')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F6F8')]),
        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#D9D9D9')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    story.extend([tb,Spacer(1,7)])
    text_sections.append('\n'.join('\t'.join(str(x) for x in row) for row in rows))
def page():story.append(PageBreak());text_sections.append('\n\n')

para('동일 BMS 입력 기반 분석 시간 비교','title')
para('측정 기록 및 후속 비교 실험 양식  |  2026년 11월 16일','small')
para('<b>동일한 BMS 관측값</b>으로 기존 ML + DT 방식과 기존 ML에 팀원 DL을 추가한 방식을 비교한다. 수명 관련 열화 조건을 찾는 성능과 전체 분석 시간을 함께 측정한다. 실제 모델 버전과 실행 흐름은 비교 전에 확정한다.')
para('현재 확인된 측정 결과','h')
table([
 ['항목','AWS 1,000건 실행 기록'],
 ['실행 결과','1,000건 완료 / 실패·중단·미완료 0건'],
 ['배치 실행 구간',f'{elapsed:,.3f}초 = 48분 7.514초'],
 ['위 시간의 경계','프로세스풀 실행 전부터 최종 집계 호출 직전까지. 사례별 계산·로컬 저장·진행 기록을 포함하며 순수 solver 시간은 분리되지 않음.'],
 ['시작 기록 → 완료 기록',f'{receipt_delta:,.3f}초 = 48분 29.641초\n11월 15일 12:14:13.128 → 13:02:42.769 KST'],
 ['하드웨어 및 병렬성','AWS m7i.2xlarge / 8 vCPU / 32 GiB / 최대 8 workers'],
 ['모델 및 실행 환경','SPMe + 평균 셀 온도 + SEI·도금 / PyBaMM 26.8.0.0 / Python 3.11.17'],
 ['결과 보존 및 서버','AWS에서 2,079,965개 시계열 행 검증. 완료 알림 발송. 11월 16일 조회 시 서버 stopped.'],
 ],[123,380])
para('시각 차이는 검증·저장 등을 포함하는 별도 관측 구간이다. 두 시간의 차이를 특정 단계의 소요시간으로 배분하지 않는다. 이번 실행은 1회이므로 반복 중앙값·p95·속도 향상률은 아직 산출하지 않는다.','small')
para('이 기록이 비교 실험에서 의미하는 것','h')
para('어제 실행은 초기 SOC·온도 등이 다른 <b>가상 상태 1,000개</b>의 전수 계산이다. 실제 BMS 입력으로 수행한 기존 ML 방식의 총시간이 아니다. 후속 비교에서는 같은 BMS에서 만든 상태와 후보 집합으로 전수 기준도 다시 맞춰야 한다.')
para('현재 low/medium/high는 온도·도금 지표 개수이며 수명 감소 순위가 아니다. 온도 지표 200건은 초기 40°C 조건과 일치하고, 도금 양성은 936건이다. 이 라벨만 잘 맞힌 결과를 수명 예측 또는 딥러닝의 이득으로 해석하지 않는다.','small')

page()
para('비교 전 고정할 입력과 실행 조건','title')
para('<b>전체 흐름 비교</b>에서는 같은 BMS 원본·관측 종료 시점을 두 방식에 넣는다. 기존 ML이 셀을 선별하고 새 DL이 시나리오를 선별한다면 역할이 다르므로 두 모델을 단순 대체 관계로 취급하지 않는다.')
para('<b>DL 선별 효과만 비교</b>할 때는 앞단 ML이 고른 셀, 상태 스냅샷, 후보 집합을 공통으로 고정한다. 데이터는 셀·이력 단위로 학습·검증·최종 평가를 분리하며, 평가 정답은 선별기에 제공하지 않는다.')
table([
 ['공통 기록 항목','작성 값 또는 확인 사항'],
 ['실험 식별','experiment_id: __________  /  paired_run_id: __________'],
 ['BMS 원본','파일명·버전: __________\nSHA256: ______________________________________________'],
 ['관측 범위','cell_id: __________  /  state_id: __________\n관측 시작·종료 시각 및 시간대: __________________________'],
 ['동일 입력','전압·전류·온도·SOC·이력의 단위, 샘플링, 결측 처리, 전처리 코드 해시와 모델에 전달한 입력 해시를 기록한다.'],
 ['동일 초기 상태','같은 BMS 정보로 동일한 상태 추정·보정을 수행한다. 상태 스냅샷과 파라미터 해시를 공유한다. 미관측 내부 상태는 가정과 불확실성을 기록한다.'],
 ['동일 후보 집합','candidate_pool_id·SHA256, 후보 수, 미래 운용 조건, 평가 기간을 고정한다. 선택된 후보 ID와 중복 제거 후 실행 ID를 저장한다.'],
 ['동일 물리 계산','모델 코드·파라미터·옵션·격자·허용오차·중단 조건·판정 기준을 고정한다. 새 방식만 DFN으로 바꾸면 모델 효과가 섞인다.'],
 ['동일 실행 환경','CPU/GPU·메모리·OS·패키지·workers·스레드 수, 캐시·초기 로딩, cold/warm 구분을 기록한다.'],
 ['방법별 차이','ML/DL 모델·가중치 버전, 선별 정책·임계값·통신 횟수·불확실성 계산·랜덤 시드·예산·종료 조건을 기록한다.'],
 ],[123,380])
para('시간 측정 범위','h')
para('<b>온라인 총시간</b>은 BMS 입력을 읽기 시작한 시점부터 최종 분석 결과가 준비된 시점까지 단조 시계로 잰다. 전처리·상태 준비, AI 추론, 후보 선별·소통, 실제 DT 실행, 결과 저장을 별도 구간으로 함께 기록한다.')
para('병렬 작업의 개별 실행시간 합과 사용자가 기다린 wall time은 다르다. 겹친 구간을 더해 총시간으로 삼지 않는다. 캐시에서 정답을 조회한 경우 실제 DT 실행과 구분한다. 학습 데이터 생성·모델 학습 비용은 온라인 시간과 분리하고, 필요할 때만 예상 사용 횟수로 나눈 상각 비용을 함께 제시한다.','small')

page()
para('동일 BMS 비교 결과 기록 양식','title')
para('아래는 후속 짝비교 실험용이다. 어제의 가상 상태 1,000건 시간을 이 표에 기존 ML 성능으로 옮겨 쓰지 않는다. 모든 시간 단위는 초이며 아직 측정하지 않은 값은 미측정으로 둔다.')
table([
 ['측정 항목','전수 DT 기준','기존 ML + DT','기존 ML + DL + DT'],
 ['실험·입력 해시 일치','미확정','미확정','미확정'],
 ['전처리·상태 준비 시간','미측정','미측정','미측정'],
 ['AI 추론 시간','해당 없음','미측정','미측정'],
 ['선별·소통 시간','해당 없음','미측정','미측정'],
 ['DT 구간 wall time','미측정','미측정','미측정'],
 ['결과 저장 시간','미측정','미측정','미측정'],
 ['온라인 총 wall time','미측정','미측정','미측정'],
 ['학습 시간 및 생성 비용','해당 없음','미측정','미측정'],
 ['고유 DT 호출 수 / 후보 수','미측정','미측정','미측정'],
 ['원인별 발견률·누락 수','미측정','미측정','미측정'],
 ['연속 열화량 오차','미측정','미측정','미측정'],
 ['반복 횟수·중앙값·p95','미측정','미측정','미측정'],
 ],[155,109,119.5,119.5])
para('성능과 시간의 판정 규칙','h')
para('1. SEI 손실 증가량·비활성 리튬 증가량 등 목표와 단위, 사건 임계값, 평가 기간을 실험 전에 정한다. 단기 지표와 실제 수명/RUL 정답은 구분한다. 원인별·복합 열화별 결과를 나눠 기록한다.')
para('2. 발견률은 “전수 기준에서 양성인 후보 중 선택해 실제 계산으로 확인한 후보 비율”인지, “계산하지 않은 후보까지 예측한 분류 recall”인지 명시한다. 두 정의를 섞지 않는다. 누락 수와 정밀도도 함께 기록한다.')
para('3. 같은 발견 성능에서 온라인 총시간·DT 호출 수가 줄었는지, 또는 같은 시간 예산에서 더 많이 발견했는지 비교한다. 반복 순서를 섞고 동일한 반복 조건을 짝지어 비교한다. 허용 누락률·반복 횟수·통과 기준은 결과를 보기 전에 확정한다.')
para('근거와 재현 기록','h')
para('실측 근거: dataset_report.json의 wall_seconds, started.json, completion.json, retrieval_receipt.json. 라벨 한계: independent_data_audit.json. 원본 결과는 2026-11-15 batch-v1로 식별한다. 현재 문서에는 Bosch 특정 구현의 재현 성능이나 미실행 ML/DL의 시간을 기입하지 않았다.','small')

def footer(c,doc):
    c.saveState();c.setFont('Malgun',8);c.setFillColor(colors.HexColor('#666666'))
    c.drawString(46,27,'배터리 수명 관련 열화 분석 비교')
    c.drawRightString(A4[0]-46,27,f'{doc.page} / 3');c.restoreState()
pdf=OUT/'시간비교_보고서.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=39,bottomMargin=43,
                      title='동일 BMS 입력 기반 분석 시간 비교',author='캡스톤 프로젝트')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'시간비교_보고서_작성양식.txt').write_text('\n\n'.join(text_sections),encoding='utf-8-sig')
r=PdfReader(pdf)
assert len(r.pages)==3,len(r.pages)
assert all('동일' in p.extract_text() or '시간' in p.extract_text() for p in r.pages)
print(json.dumps({'pdf':str(pdf),'pages':len(r.pages),'batch_wall_seconds':elapsed,'receipt_delta_seconds':receipt_delta},ensure_ascii=False))
